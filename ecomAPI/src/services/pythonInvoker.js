import { spawn } from 'child_process';
import path from 'path';

const ROOT = path.resolve(__dirname, '../../..');
const PYTHON = process.env.PYTHON_BIN || 'python';
const SCRIPT = path.join(ROOT, 'models', 'recommend_api.py');

function runPythonInference(payload, { timeoutMs = 15000 } = {}) {
  return new Promise((resolve) => {
    try {
      const ps = spawn(PYTHON, ['-u', SCRIPT], { cwd: ROOT });
      let out = '';
      let err = '';
      const timer = setTimeout(() => {
        try { ps.kill('SIGKILL'); } catch {}
        resolve({ ok: false, error: 'timeout' });
      }, timeoutMs);

      ps.stdout.on('data', (d) => { out += d.toString(); });
      ps.stderr.on('data', (d) => { err += d.toString(); });
      ps.on('close', () => {
        clearTimeout(timer);
        try {
          const parsed = JSON.parse(out || '{}');
          resolve(parsed);
        } catch (e) {
          resolve({ ok: false, error: 'invalid_json', raw: out, stderr: err });
        }
      });
      ps.stdin.write(JSON.stringify(payload || {}));
      ps.stdin.end();
    } catch (e) {
      resolve({ ok: false, error: e?.message || String(e) });
    }
  });
}

export default { runPythonInference };
