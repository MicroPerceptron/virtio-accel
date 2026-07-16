//! No-std reference driver for portable virtio-accel protocol 1.0.
//!
//! Callers provide complete driver-owned chains. The client writes exact request frames into those
//! chains, tracks bounded in-flight ownership, and validates all device-written bytes after used
//! reclamation. It names no VMM, guest-memory library, operating system, or device backend.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod client;
mod config;
mod operation;
mod types;

pub use client::{
    ClientHealth, ClientInitError, Completion, GuestClient, Pending, PumpResult, QueueSetupError,
    RequestPoll, StaleOperation, StartError, StartErrorKind, StartResult,
};
pub use config::{GuestConfig, GuestConfigError};
pub use operation::{
    AllocateBuffer, CancelEvent, CreateContext, CreateExecutionQueue, DestroyContext, DestroyEvent,
    DestroyExecutionQueue, FreeBuffer, GetDeviceInfo, LoadProgram, Operation, PollEvent,
    ReadBuffer, ResponseError, Submit, UnloadProgram, WriteBuffer,
};
pub use types::{
    AccessMode, Binding, Buffer, BufferDesc, BufferRange, BufferUsage, Context, DeviceInfo,
    DeviceInfoError, Event, EventState, ExecutionQueue, FailureDisposition, MemoryDomain, Program,
    ProgramDesc, ReadBufferOutput, SubmissionOutcome, ValueError,
};
