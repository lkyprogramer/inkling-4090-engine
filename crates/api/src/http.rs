// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! A minimal HTTP/1.1 layer on `std::net`: one request per connection (`Connection: close`),
//! `Content-Length` bodies, plain responses and Server-Sent Events.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

/// Largest request body accepted (prompts are text; 64 MB is far beyond any context).
const MAX_BODY: usize = 64 << 20;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Reads one request. `Ok(None)` when the client closed the connection before sending one.
pub fn read_request(stream: &TcpStream) -> io::Result<Option<Request>> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let (method, target) = match (parts.next(), parts.next()) {
        (Some(m), Some(t)) => (m.to_string(), t.to_string()),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad request line")),
    };
    let path = target.split('?').next().unwrap_or("").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad content-length"))?;
            } else if k.trim().eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked") {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "chunked request bodies are not supported"));
            }
        }
    }
    if len > MAX_BODY {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "request body too large"));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "",
    }
}

pub fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> io::Result<()> {
    let head = format!("HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", reason(status), body.len());
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

pub fn respond_json(stream: &mut TcpStream, status: u16, v: &serde_json::Value) -> io::Result<()> {
    respond(stream, status, "application/json", v.to_string().as_bytes())
}

/// OpenAI-style error body.
pub fn respond_error(stream: &mut TcpStream, status: u16, msg: &str) -> io::Result<()> {
    let kind = if status >= 500 { "server_error" } else { "invalid_request_error" };
    respond_json(stream, status, &serde_json::json!({ "error": { "message": msg, "type": kind, "code": status } }))
}

pub fn sse_start(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n")?;
    stream.flush()
}

pub fn sse_send(stream: &mut TcpStream, data: &str) -> io::Result<()> {
    stream.write_all(b"data: ")?;
    stream.write_all(data.as_bytes())?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}
