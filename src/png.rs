//! Minimal PNG encoder, used to write captured frames to disk.
//!
//! Deliberately dependency-free: capture is a debug/verification path, and a
//! stored-deflate zlib stream keeps the writer to ~80 lines that anyone (or any
//! tool) can read. The output is a normal 8-bit RGBA PNG that any viewer or
//! image library opens. Compression is nil, so a 2560x1440 frame is ~14 MB;
//! that is fine for a handful of diagnostic captures.

use std::io::{self, Write};
use std::path::Path;

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
/// PNG stored-deflate blocks cap the payload at 65535 bytes each.
const MAX_STORED: usize = 0xffff;

/// Encode 8-bit RGBA pixels (row-major, top row first) as a PNG file.
pub fn encode(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let stride = width as usize * 4;
    assert_eq!(
        rgba.len(),
        stride * height as usize,
        "pixel buffer must be width*height*4 bytes"
    );

    let mut out = Vec::with_capacity(rgba.len() + rgba.len() / MAX_STORED * 5 + 128);
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, no interlace
    chunk(&mut out, b"IHDR", &ihdr);

    // Raw scanlines with filter type 0 (none).
    let mut raw = Vec::with_capacity(height as usize * (stride + 1));
    for row in rgba.chunks_exact(stride) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Write 8-bit RGBA pixels to `path` as a PNG.
pub fn write(path: &Path, width: u32, height: u32, rgba: &[u8]) -> io::Result<()> {
    let bytes = encode(width, height, rgba);
    std::fs::File::create(path)?.write_all(&bytes)
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// zlib container around deflate "stored" blocks, no compression at all.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / MAX_STORED * 5 + 16);
    out.extend_from_slice(&[0x78, 0x01]); // deflate, 32 KiB window
    let blocks = raw.chunks(MAX_STORED).collect::<Vec<_>>();
    let block_count = blocks.len().max(1);
    for (index, block) in blocks.iter().enumerate() {
        let last = index + 1 == block_count;
        out.push(u8::from(last)); // BFINAL, BTYPE = 00 (stored)
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    if blocks.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Self(0xffff_ffff)
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.0 ^= byte as u32;
            for _ in 0..8 {
                let mask = (self.0 & 1).wrapping_neg();
                self.0 = (self.0 >> 1) ^ (0xedb8_8320 & mask);
            }
        }
    }

    fn finish(self) -> u32 {
        !self.0
    }
}
