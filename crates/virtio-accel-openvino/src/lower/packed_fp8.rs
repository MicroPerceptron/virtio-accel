//! Exact FP8 activation conversion with byte-preserving graph boundaries.
use super::*;

struct Encoder<'a>(&'a mut IrBuilder);
impl Encoder<'_> {
    fn op(
        &mut self,
        kind: &str,
        data: &str,
        ins: &[(PortRef, &[i64])],
        dtype: OvElement,
        shape: &[i64],
    ) -> PortRef {
        let name = format!("fp8_{}", self.0.next_layer);
        self.0.emit_layer(
            kind,
            if kind == "Gather" { "opset8" } else { "opset1" },
            &name,
            data,
            ins,
            &[(dtype, shape)],
        )[0]
    }
    fn constant(
        &mut self,
        dtype: OvElement,
        shape: &[i64],
        bytes: &[u8],
    ) -> Result<PortRef, LoweringError> {
        let name = format!("fp8_{}", self.0.next_layer);
        self.0.emit_const(&name, dtype, shape, bytes)
    }
    fn int(&mut self, x: i32) -> Result<PortRef, LoweringError> {
        self.constant(OvElement::I32, &[], &x.to_le_bytes())
    }
    fn float(&mut self, x: f32) -> Result<PortRef, LoweringError> {
        self.constant(OvElement::F32, &[], &x.to_le_bytes())
    }
    fn convert(&mut self, x: PortRef, dtype: OvElement, s: &[i64]) -> PortRef {
        self.op(
            "Convert",
            &format!("destination_type=\"{}\"", dtype.element_type()),
            &[(x, s)],
            dtype,
            s,
        )
    }
    fn binary(
        &mut self,
        kind: &str,
        x: (PortRef, &[i64]),
        y: (PortRef, &[i64]),
        d: OvElement,
        s: &[i64],
    ) -> PortRef {
        self.op(kind, "auto_broadcast=\"numpy\"", &[x, y], d, s)
    }
    fn select(
        &mut self,
        c: PortRef,
        a: (PortRef, &[i64]),
        b: (PortRef, &[i64]),
        d: OvElement,
        s: &[i64],
    ) -> PortRef {
        self.op("Select", "auto_broadcast=\"numpy\"", &[(c, s), a, b], d, s)
    }
    fn lookup(
        &mut self,
        table: &[f32],
        index: PortRef,
        s: &[i64],
    ) -> Result<PortRef, LoweringError> {
        let dims = [table.len() as i64];
        let table = self.constant(
            OvElement::F32,
            &dims,
            &table
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let axis = self.int(0)?;
        Ok(self.op(
            "Gather",
            "batch_dims=\"0\"",
            &[(table, &dims), (index, s), (axis, &[])],
            OvElement::F32,
            s,
        ))
    }
    fn decode(&mut self, x: PortRef, dtype: DType, s: &[i64]) -> Result<PortRef, LoweringError> {
        let x = self.convert(x, OvElement::I32, s);
        let offset = self.int(128)?;
        let index = self.binary("Add", (x, s), (offset, &[]), OvElement::I32, s);
        let table = (0..256)
            .map(|i| decode((i as u8).wrapping_add(128), dtype))
            .collect::<Vec<_>>();
        self.lookup(&table, index, s)
    }
    fn encode(&mut self, x: PortRef, dtype: DType, s: &[i64]) -> Result<PortRef, LoweringError> {
        let x = self.convert(x, OvElement::F32, s);
        let mag = self.op("Abs", "", &[(x, s)], OvElement::F32, s);
        let mut code = self.int(0)?;
        let max_code = if dtype == DType::FP8E4M3 { 127 } else { 124 };
        let mut thresholds = vec![0.0; 128];
        let mut even = vec![0.0; 128];
        for i in 1..128 {
            thresholds[i] = if i >= max_code {
                if dtype == DType::FP8E4M3 {
                    464.0
                } else if i == 124 {
                    61440.0
                } else {
                    f32::INFINITY
                }
            } else {
                (decode((i - 1) as u8, dtype) + decode(i as u8, dtype)) * 0.5
            };
            even[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        let zero = self.float(0.0)?;
        // Direct binary32 round-to-nearest-even. No binary16 intermediate.
        for step in [64, 32, 16, 8, 4, 2, 1] {
            let old_shape = if step == 64 { &[][..] } else { s };
            let amount = self.int(step)?;
            let trial = self.binary(
                "Add",
                (code, old_shape),
                (amount, &[]),
                OvElement::I32,
                old_shape,
            );
            let threshold = self.lookup(&thresholds, trial, old_shape)?;
            let greater = self.binary(
                "Greater",
                (mag, s),
                (threshold, old_shape),
                OvElement::Bool,
                s,
            );
            let equal = self.binary(
                "Equal",
                (mag, s),
                (threshold, old_shape),
                OvElement::Bool,
                s,
            );
            let parity = self.lookup(&even, trial, old_shape)?;
            let is_even = self.binary(
                "Greater",
                (parity, old_shape),
                (zero, &[]),
                OvElement::Bool,
                old_shape,
            );
            let tie = self.binary(
                "LogicalAnd",
                (equal, s),
                (is_even, old_shape),
                OvElement::Bool,
                s,
            );
            let up = self.binary("LogicalOr", (greater, s), (tie, s), OvElement::Bool, s);
            code = self.select(up, (trial, old_shape), (code, old_shape), OvElement::I32, s);
        }
        let infinity = self.float(f32::INFINITY)?;
        let is_inf = self.binary("Equal", (mag, s), (infinity, &[]), OvElement::Bool, s);
        let overflow = self.int(max_code as i32)?;
        code = self.select(is_inf, (overflow, &[]), (code, s), OvElement::I32, s);
        let number = self.binary("Equal", (x, s), (x, s), OvElement::Bool, s);
        let nan = self.int(if dtype == DType::FP8E4M3 { 127 } else { 126 })?;
        code = self.select(number, (code, s), (nan, &[]), OvElement::I32, s);
        let negative = self.binary("Less", (x, s), (zero, &[]), OvElement::Bool, s);
        let one = self.float(1.0)?;
        let inverse = self.binary("Divide", (one, &[]), (x, s), OvElement::F32, s);
        let neg_zero = self.binary("Less", (inverse, s), (zero, &[]), OvElement::Bool, s);
        let sign = self.binary(
            "LogicalOr",
            (negative, s),
            (neg_zero, s),
            OvElement::Bool,
            s,
        );
        let signbit = self.int(128)?;
        let signed = self.binary("Subtract", (code, s), (signbit, &[]), OvElement::I32, s);
        let bits = self.select(sign, (signed, s), (code, s), OvElement::I32, s);
        Ok(self.convert(bits, OvElement::I8, s))
    }
}
fn decode(x: u8, d: DType) -> f32 {
    if d == DType::FP8E4M3 {
        virtio_accel_tosa::fp8e4m3_to_f32(x)
    } else {
        virtio_accel_tosa::fp8e5m2_to_f32(x)
    }
}
pub(super) fn cast(
    b: &mut IrBuilder,
    x: PortRef,
    input: DType,
    output: DType,
    s: &[i64],
) -> Result<PortRef, LoweringError> {
    let mut e = Encoder(b);
    let fp8 = |d| matches!(d, DType::FP8E4M3 | DType::FP8E5M2);
    if input == output {
        return Ok(e.convert(x, e.0.element(output)?, s));
    }
    let x = if fp8(input) {
        e.decode(x, input, s)?
    } else {
        x
    };
    if fp8(output) {
        e.encode(x, output, s)
    } else {
        Ok(e.convert(x, OvElement::for_dtype(output)?, s))
    }
}
