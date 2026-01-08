import { spawn } from 'child_process';
import path from 'path';
import os from 'os';

const ROOT = path.resolve(__dirname, '../../..');

// Tự động phát hiện Python
const DEFAULT_PYTHON =
  process.platform === 'win32'
    ? 'python.exe'
    : 'python3'; // Linux/macOS thường dùng python3

const PYTHON = process.env.PYTHON_BIN || DEFAULT_PYTHON;
const SCRIPT = path.join(ROOT, 'models', 'recommend_api.py');

function runPythonInference(payload, { timeoutMs = 120000 } = {}) {
  return new Promise((resolve) => {
    try {
      // Môi trường cho tiến trình Python
      const env = {
        ...process.env, // Giữ nguyên môi trường gốc
        PYTHONUNBUFFERED: '1',
        DB_HOST: process.env.DB_HOST || 'localhost',
        DB_USERNAME: process.env.DB_USERNAME || 'root',
        DB_PASSWORD: process.env.DB_PASSWORD || '',
        DB_DATABASE_NAME: process.env.DB_DATABASE_NAME || 'ecom',
        DB_PORT: process.env.DB_PORT || '3306'
      };

      console.log(`[PYTHON] Environment setup:`, {
        DB_HOST: env.DB_HOST,
        DB_USERNAME: env.DB_USERNAME,
        DB_DATABASE_NAME: env.DB_DATABASE_NAME,
        working_dir: ROOT,
        python_bin: PYTHON
      });

      const ps = spawn(PYTHON, ['-u', SCRIPT], {
        cwd: ROOT,
        env,
        shell: process.platform === 'win32' // để chạy được trên Windows
      });

      let out = '';
      let err = '';

      const timer = setTimeout(() => {
        console.log(`[PYTHON] Timeout (${timeoutMs}ms) — killing process`);
        try {
          if (process.platform === 'win32') {
            spawn('taskkill', ['/pid', ps.pid, '/f', '/t']);
          } else {
            ps.kill('SIGKILL');
          }
        } catch {}
        resolve({ ok: false, error: 'timeout', raw: out, stderr: err });
      }, timeoutMs);

      ps.stdout.on('data', (d) => { out += d.toString('utf8'); });
      ps.stderr.on('data', (d) => { err += d.toString('utf8'); });

      ps.on('close', (code) => {
        clearTimeout(timer);
        console.log(`[PYTHON] Process exited with code ${code}`);
        console.log(`[PYTHON] Raw stdout:`, out);
        console.log(`[PYTHON] Raw stderr:`, err);

        if (code !== 0) {
          resolve({ ok: false, error: `exit_code_${code}`, raw: out, stderr: err });
          return;
        }

        if (!out || !out.trim()) {
          resolve({ ok: false, error: 'no_output', raw: out, stderr: err });
          return;
        }

        try {
          const parsed = JSON.parse(out.trim());
          resolve(parsed);
        } catch (e) {
          resolve({ ok: false, error: 'invalid_json', raw: out, stderr: err });
        }
      });

      // Gửi input cho Python
      const inputData = JSON.stringify(payload || {});
      console.log(`[PYTHON] Sending input data: ${inputData}`);
      ps.stdin.write(inputData + os.EOL);
      ps.stdin.end();
    } catch (e) {
      resolve({ ok: false, error: e?.message || String(e) });
    }
  });
}

export default { runPythonInference };
