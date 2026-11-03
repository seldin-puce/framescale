//! Read PyTorch `.pth` state dicts (zip archive + pickle) **without executing
//! anything**. The pickle is interpreted, not loaded: only the opcodes
//! `torch.save` emits are understood, and only an allow-list of globals
//! (`OrderedDict`, tensor storages, `_rebuild_tensor_v2`, `_rebuild_parameter`)
//! may be named. Anything else is an error. Tensors come out as f32.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float,
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    Storage(String, Dtype),
    Tensor {
        key: String,
        dtype: Dtype,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    },
    Mark,
}

fn int(v: &V) -> Result<i64> {
    match v {
        V::Int(i) => Ok(*i),
        V::Bool(b) => Ok(i64::from(*b)),
        _ => bail!("expected an integer, got {v:?}"),
    }
}

fn ints(v: &V) -> Result<Vec<usize>> {
    match v {
        V::Tuple(xs) | V::List(xs) => xs
            .iter()
            .map(|x| usize::try_from(int(x)?).context("negative size"))
            .collect(),
        _ => bail!("expected a tuple of integers, got {v:?}"),
    }
}

fn storage_dtype(name: &str) -> Option<Dtype> {
    match name {
        "torch FloatStorage" => Some(Dtype::F32),
        "torch HalfStorage" => Some(Dtype::F16),
        "torch BFloat16Storage" => Some(Dtype::Bf16),
        _ => None,
    }
}

const ALLOWED: [&str; 6] = [
    "collections OrderedDict",
    "torch FloatStorage",
    "torch HalfStorage",
    "torch BFloat16Storage",
    "torch._utils _rebuild_tensor_v2",
    "torch._utils _rebuild_parameter",
];

/// Interpret a pickle into a value tree.
fn interpret(pk: &[u8]) -> Result<V> {
    let mut stack: Vec<V> = Vec::new();
    let mut memo: BTreeMap<u32, V> = BTreeMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = pk.get(*i..*i + n).context("truncated pickle")?;
        *i += n;
        Ok(s)
    };
    let u32le = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let pop = |stack: &mut Vec<V>| stack.pop().context("pickle stack underflow");
    let pop_mark = |stack: &mut Vec<V>| -> Result<Vec<V>> {
        let at = stack
            .iter()
            .rposition(|v| matches!(v, V::Mark))
            .context("pickle MARK missing")?;
        let items = stack.split_off(at + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = *pk.get(i).context("pickle ended without STOP")?;
        i += 1;
        match op {
            0x80 => {
                take(&mut i, 1)?; // PROTO
            }
            0x95 => {
                take(&mut i, 8)?; // FRAME
            }
            b'.' => return pop(&mut stack), // STOP
            b'(' => stack.push(V::Mark),
            b'}' => stack.push(V::Dict(Vec::new())),
            b')' => stack.push(V::Tuple(Vec::new())),
            b']' => stack.push(V::List(Vec::new())),
            b'N' => stack.push(V::None),
            0x88 => stack.push(V::Bool(true)),
            0x89 => stack.push(V::Bool(false)),
            b'K' => {
                let b = take(&mut i, 1)?;
                stack.push(V::Int(i64::from(b[0])));
            }
            b'M' => {
                let b = take(&mut i, 2)?;
                stack.push(V::Int(i64::from(u16::from_le_bytes([b[0], b[1]]))));
            }
            b'J' => {
                let b = take(&mut i, 4)?;
                stack.push(V::Int(i64::from(i32::from_le_bytes([
                    b[0], b[1], b[2], b[3],
                ]))));
            }
            0x8a => {
                // LONG1: little-endian two's complement
                let n = usize::from(take(&mut i, 1)?[0]);
                let b = take(&mut i, n)?;
                if n > 8 {
                    bail!("pickle integer too large");
                }
                let mut v: i64 = 0;
                for (k, byte) in b.iter().enumerate() {
                    v |= i64::from(*byte) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(V::Int(v));
            }
            b'G' => {
                let b = take(&mut i, 8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(b);
                let _ = f64::from_be_bytes(a); // unused by state dicts
                stack.push(V::Float);
            }
            b'X' | 0x8c => {
                // BINUNICODE (u32 length) / SHORT_BINUNICODE (u8 length)
                let n = if op == b'X' {
                    u32le(take(&mut i, 4)?) as usize
                } else {
                    usize::from(take(&mut i, 1)?[0])
                };
                let s = std::str::from_utf8(take(&mut i, n)?).context("pickle string")?;
                stack.push(V::Str(s.to_owned()));
            }
            b'c' => {
                // GLOBAL "module\nname\n"
                let rest = pk.get(i..).context("truncated GLOBAL")?;
                let mut lines = rest.splitn(3, |&b| b == b'\n');
                let module = lines.next().context("GLOBAL module")?;
                let name = lines.next().context("GLOBAL name")?;
                i += module.len() + name.len() + 2;
                let g = format!(
                    "{} {}",
                    String::from_utf8_lossy(module),
                    String::from_utf8_lossy(name)
                );
                if !ALLOWED.contains(&g.as_str()) {
                    bail!("refusing pickle global `{g}`");
                }
                stack.push(V::Global(g));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                let (V::Str(m), V::Str(n)) = (module, name) else {
                    bail!("STACK_GLOBAL needs strings");
                };
                let g = format!("{m} {n}");
                if !ALLOWED.contains(&g.as_str()) {
                    bail!("refusing pickle global `{g}`");
                }
                stack.push(V::Global(g));
            }
            b'q' => {
                let k = u32::from(take(&mut i, 1)?[0]);
                memo.insert(k, stack.last().context("BINPUT on empty stack")?.clone());
            }
            b'r' => {
                let k = u32le(take(&mut i, 4)?);
                memo.insert(
                    k,
                    stack.last().context("LONG_BINPUT on empty stack")?.clone(),
                );
            }
            0x94 => {
                let k = memo.len() as u32; // MEMOIZE
                memo.insert(k, stack.last().context("MEMOIZE on empty stack")?.clone());
            }
            b'h' => {
                let k = u32::from(take(&mut i, 1)?[0]);
                stack.push(memo.get(&k).context("BINGET of unknown memo")?.clone());
            }
            b'j' => {
                let k = u32le(take(&mut i, 4)?);
                stack.push(memo.get(&k).context("LONG_BINGET of unknown memo")?.clone());
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(V::Tuple(items));
            }
            0x85..=0x87 => {
                let n = usize::from(op - 0x84);
                if stack.len() < n {
                    bail!("pickle stack underflow");
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(V::Tuple(items));
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(V::List(l)) => l.push(v),
                    _ => bail!("APPEND to a non-list"),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(V::List(l)) => l.extend(items),
                    _ => bail!("APPENDS to a non-list"),
                }
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(V::Dict(d)) => d.push((k, v)),
                    _ => bail!("SETITEM on a non-dict"),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                let Some(V::Dict(d)) = stack.last_mut() else {
                    bail!("SETITEMS on a non-dict");
                };
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    d.push((k, v));
                }
            }
            b'b' => {
                // BUILD: object state (OrderedDict `_metadata`) is irrelevant.
                pop(&mut stack)?;
            }
            b'Q' => {
                // BINPERSID: ('storage', <Storage class>, key, location, numel)
                let pid = pop(&mut stack)?;
                let V::Tuple(p) = pid else {
                    bail!("persistent id isn't a tuple");
                };
                let (Some(V::Global(class)), Some(V::Str(key))) = (p.get(1), p.get(2)) else {
                    bail!("unexpected persistent id {p:?}");
                };
                let dtype =
                    storage_dtype(class).with_context(|| format!("storage type `{class}`"))?;
                stack.push(V::Storage(key.clone(), dtype));
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let f = pop(&mut stack)?;
                let V::Tuple(args) = args else {
                    bail!("REDUCE arguments aren't a tuple");
                };
                let V::Global(g) = f else {
                    bail!("REDUCE on a non-global");
                };
                let v = match g.as_str() {
                    "collections OrderedDict" => V::Dict(Vec::new()),
                    "torch._utils _rebuild_tensor_v2" => {
                        let (Some(V::Storage(key, dtype)), Some(offset), Some(shape), Some(stride)) =
                            (args.first(), args.get(1), args.get(2), args.get(3))
                        else {
                            bail!("unexpected _rebuild_tensor_v2 arguments");
                        };
                        V::Tensor {
                            key: key.clone(),
                            dtype: *dtype,
                            offset: usize::try_from(int(offset)?).context("offset")?,
                            shape: ints(shape)?,
                            stride: ints(stride)?,
                        }
                    }
                    "torch._utils _rebuild_parameter" => args
                        .into_iter()
                        .next()
                        .context("_rebuild_parameter without data")?,
                    _ => bail!("refusing to call `{g}`"),
                };
                stack.push(v);
            }
            _ => bail!("unsupported pickle opcode 0x{op:02x} at byte {}", i - 1),
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Load every tensor of a `.pth` state dict. A checkpoint that wraps the
/// weights (`{"params": …}` / `{"params_ema": …}`) is unwrapped; `params_ema`
/// wins when both exist, as in Real-ESRGAN's own loader.
pub fn load(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut zip =
        zip::ZipArchive::new(file).with_context(|| format!("{} isn't a zip", path.display()))?;
    let names = zip
        .file_names()
        .map(|n| n.map(std::borrow::Cow::into_owned))
        .collect::<Result<Vec<String>, _>>()?;
    let pkl = names
        .iter()
        .find(|n| n.ends_with("/data.pkl") || n.as_str() == "data.pkl")
        .context("no data.pkl in the archive")?
        .clone();
    let prefix = pkl.trim_end_matches("data.pkl").to_owned();
    let mut pk = Vec::new();
    zip.by_name(&pkl)?.read_to_end(&mut pk)?;
    let root = interpret(&pk)?;

    let dict = |v: V| match v {
        V::Dict(d) => Ok(d),
        other => bail!("expected a dict, got {other:?}"),
    };
    let mut top = dict(root)?;
    let wrapped = ["params_ema", "params"].iter().find_map(|k| {
        top.iter()
            .position(|(key, _)| matches!(key, V::Str(s) if s == k))
    });
    let entries = match wrapped {
        Some(at) => dict(top.swap_remove(at).1)?,
        None => top,
    };

    let mut out = BTreeMap::new();
    for (k, v) in entries {
        let V::Str(name) = k else { continue };
        let V::Tensor {
            key,
            dtype,
            offset,
            shape,
            stride,
        } = v
        else {
            continue;
        };
        let numel: usize = shape.iter().product();
        let mut contiguous = vec![1usize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            contiguous[d] = contiguous[d + 1] * shape[d + 1];
        }
        if numel > 1 && stride != contiguous {
            bail!("{name}: non-contiguous tensor (stride {stride:?})");
        }
        let mut raw = Vec::new();
        zip.by_name(&format!("{prefix}data/{key}"))
            .with_context(|| format!("{name}: storage {key} missing"))?
            .read_to_end(&mut raw)?;
        let width = if dtype == Dtype::F32 { 4 } else { 2 };
        let bytes = raw
            .get(offset * width..(offset + numel) * width)
            .with_context(|| format!("{name}: storage too short"))?;
        let data: Vec<f32> = match dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16_to_f32(u16::from_le_bytes(*b)))
                .collect(),
            Dtype::Bf16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect(),
        };
        out.insert(name, Tensor { shape, data });
    }
    if out.is_empty() {
        bail!("{}: no tensors found", path.display());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    #[test]
    fn refuses_other_globals() {
        // GLOBAL os\nsystem\n — the classic pickle exploit — must not get past the reader.
        let pk = b"\x80\x02cos\nsystem\nq\x00X\x02\x00\x00\x00idq\x01\x85q\x02Rq\x03.";
        let err = interpret(pk).expect_err("os.system");
        assert!(err.to_string().contains("refusing pickle global"), "{err}");
    }

    #[test]
    fn reads_a_state_dict() {
        // {"params": OrderedDict(w=tensor(FloatStorage "0", offset 1, shape (2,), stride (1,)))}
        let pk: &[u8] = b"\x80\x02}q\x00X\x06\x00\x00\x00paramsq\x01ccollections\nOrderedDict\nq\x02)Rq\x03X\x01\x00\x00\x00wq\x04ctorch._utils\n_rebuild_tensor_v2\nq\x05((X\x07\x00\x00\x00storageq\x06ctorch\nFloatStorage\nq\x07X\x01\x00\x00\x000q\x08X\x03\x00\x00\x00cpuq\tK\x03tq\nQK\x01K\x02\x85q\x0bK\x01\x85q\x0c\x89h\x02)Rq\rtq\x0eRq\x0fss.";
        let v = interpret(pk).expect("interpret");
        let V::Dict(top) = v else { panic!("dict") };
        let V::Dict(params) = &top[0].1 else {
            panic!("params")
        };
        let V::Tensor {
            key, offset, shape, ..
        } = &params[0].1
        else {
            panic!("tensor")
        };
        assert_eq!(
            (key.as_str(), *offset, shape.as_slice()),
            ("0", 1, &[2usize][..])
        );
    }
}
