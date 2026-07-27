//! Atomic preflight for one transport-neutral command frame.
//!
//! This is the only byte-oriented entry point needed by the command engine. It validates the
//! flattened chain shape, confirms that the byte ports match the advertised totals, decodes the
//! complete request, and writes recoverable protocol errors before returning. Only
//! [`FramePreflight::Ready`] contains a request that semantic dispatch may act on.

use virtio_accel_core::{ByteSink, ByteSource};
use virtio_accel_proto::StatusCode;

use crate::{
    ChainLayoutError, ChainRegion, DecodedRequest, FrameDecodeError, FrameDecoder,
    ResponseWriteError, ResponseWriter, UnrecoverableDecodeError, validate_chain_layout,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnusableFrame {
    ChainLayout(ChainLayoutError),
    Request(UnrecoverableDecodeError),
    InsufficientResponse {
        request_id: u64,
        required: u64,
        available: u64,
    },
}

#[derive(Debug)]
pub enum FramePreflight<'a> {
    /// The complete frame is validated and may proceed to semantic dispatch.
    Ready(DecodedRequest<'a>),
    /// A recoverable protocol error was written to the response port.
    Rejected {
        request_id: u64,
        status: StatusCode,
        used: u32,
    },
    /// No response bytes were written and the chain must complete with used length zero.
    Unusable(UnusableFrame),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramePreflightError {
    ResponseWrite(ResponseWriteError),
}

/// Validate one complete chain before semantic state or a backend can be reached.
///
/// The function is allocation-free except for the bounded `SUBMIT` binding allocation performed
/// by [`FrameDecoder`]. `Ready` leaves the response untouched. `Rejected` initializes exactly one
/// 16-byte error header. `Unusable` leaves the response untouched.
pub fn preflight_command_frame<'a>(
    decoder: &FrameDecoder,
    regions: &[ChainRegion],
    request: &'a dyn ByteSource,
    response: &mut dyn ByteSink,
) -> Result<FramePreflight<'a>, FramePreflightError> {
    let layout = match validate_chain_layout(regions, decoder.limits().max_chain_descriptors()) {
        Ok(layout) => layout,
        Err(error) => {
            return Ok(FramePreflight::Unusable(UnusableFrame::ChainLayout(error)));
        }
    };
    if let Err(error) = layout.validate_port_lengths(request.len(), response.len()) {
        return Ok(FramePreflight::Unusable(UnusableFrame::ChainLayout(error)));
    }

    match decoder.decode(request, response.len()) {
        Ok(request) => Ok(FramePreflight::Ready(request)),
        Err(FrameDecodeError::Protocol { request_id, status }) => {
            let used = ResponseWriter::new(response, decoder.limits().max_response_bytes())
                .write_empty(status, request_id)
                .map_err(FramePreflightError::ResponseWrite)?;
            Ok(FramePreflight::Rejected {
                request_id,
                status,
                used,
            })
        }
        Err(FrameDecodeError::Unrecoverable(error)) => {
            Ok(FramePreflight::Unusable(UnusableFrame::Request(error)))
        }
        Err(FrameDecodeError::InsufficientResponse {
            request_id,
            required,
            available,
        }) => Ok(FramePreflight::Unusable(
            UnusableFrame::InsufficientResponse {
                request_id,
                required,
                available,
            },
        )),
    }
}
