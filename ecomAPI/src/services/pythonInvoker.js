import { spawn } from 'child_process';
import path from 'path';

const ROOT = path.resolve(__dirname, '../../..');
const PYTHON =
  process.env.PYTHON_BIN ||
  '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3';
const SCRIPT = path.join(ROOT, 'models', 'recommend_api.py');

function runPythonInference(payload, { timeoutMs = 15000 } = {}) {
  return new Promise((resolve) => {
    const ps = spawn(PYTHON, ['-u', SCRIPT], { cwd: ROOT });

    let out = '';
    let err = '';

    const timer = setTimeout(() => {
      ps.kill('SIGKILL');
      resolve({ ok: false, error: 'timeout' });
    }, timeoutMs);

    ps.on('error', (e) => {
      clearTimeout(timer);
      resolve({ ok: false, error: 'spawn_failed', message: e.message });
    });

    ps.stdout.on('data', (d) => (out += d.toString()));
    ps.stderr.on('data', (d) => (err += d.toString()));

    ps.on('close', () => {
      clearTimeout(timer);
      try {
        resolve(JSON.parse(out || '{}'));
      } catch {
        resolve({ ok: false, error: 'invalid_json', raw: out, stderr: err });
      }
    });

    ps.stdin.write(JSON.stringify(payload || {}));
    ps.stdin.end();
  });
}

export default { runPythonInference };
