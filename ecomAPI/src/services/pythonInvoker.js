import { spawn } from 'child_process';
import path from 'path';

const ROOT = path.resolve(__dirname, '../../..');
const PYTHON = process.env.PYTHON_BIN || 'C:\\Users\\Admin\\AppData\\Local\\Programs\\Python\\Python310\\python.exe';
const SCRIPT = path.join(ROOT, 'models', 'recommend_api_trained.py');

function runPythonInference(payload, { timeoutMs = 120000 } = {}) { // 2 minutes
  return new Promise((resolve) => {
    try {
      // Create clean environment for Python process
      const env = {
        // Copy essential environment variables
        PATH: process.env.PATH,
        PYTHONPATH: process.env.PYTHONPATH || '',
        PYTHONUNBUFFERED: '1',
        // Database connection
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
        working_dir: ROOT
      });

      // Use the full recommendation system
      const scriptPath = SCRIPT;

      const ps = spawn(PYTHON, ['-u', scriptPath], { cwd: ROOT, env });
      let out = '';
      let err = '';
      const timer = setTimeout(() => {
        try { ps.kill('SIGKILL'); } catch {}
        resolve({ ok: false, error: 'timeout', raw: out, stderr: err });
      }, timeoutMs);

      ps.stdout.on('data', (d) => { out += d.toString(); });
      ps.stderr.on('data', (d) => { err += d.toString(); });

      ps.on('close', (code) => {
        clearTimeout(timer);

        console.log(`[PYTHON] Process completed with exit code: ${code}`);
        console.log(`[PYTHON] Raw stdout: '${out}'`);
        console.log(`[PYTHON] Raw stderr: '${err}'`);

        if (code !== 0) {
          console.log(`[PYTHON] Process failed with exit code ${code}`);
          resolve({ ok: false, error: `exit_code_${code}`, raw: out, stderr: err });
          return;
        }

        if (!out || !out.trim()) {
          console.log(`[PYTHON] No stdout output received`);
          resolve({ ok: false, error: 'no_output', raw: out, stderr: err });
          return;
        }

        try {
          const parsed = JSON.parse(out.trim());
          console.log(`[PYTHON] Successfully parsed JSON`);
          resolve(parsed);
        } catch (e) {
          console.log(`[PYTHON] JSON parse error: ${e.message}`);
          resolve({ ok: false, error: 'invalid_json', raw: out, stderr: err });
        }
      });
      const inputData = JSON.stringify(payload || {});
      console.log(`[PYTHON] Sending input data: ${inputData}`);
      ps.stdin.write(inputData + '\n'); // Add newline to ensure complete input
      ps.stdin.end();
    } catch (e) {
      resolve({ ok: false, error: e?.message || String(e) });
    }
  });
}

export default { runPythonInference };
