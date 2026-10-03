import fs from 'fs';
import path from 'path';
const dir = process.argv[2] || '_posts';
const filter = process.argv[3] || 'flow-matching';
const outPath = process.argv[4] || '/tmp/mermaid_cases.json';
const out = [];
for (const f of fs.readdirSync(dir).filter(x => x.endsWith('.md') && (filter === '*' || x.includes(filter)))) {
  const lines = fs.readFileSync(path.join(dir, f), 'utf8').split('\n');
  let i = 0, idx = 0;
  while (i < lines.length) {
    if (lines[i].trim().startsWith('```mermaid')) {
      let j = i + 1;
      const body = [];
      while (j < lines.length && lines[j].trim() !== '```') { body.push(lines[j]); j += 1; }
      out.push({ file: f, n: ++idx, code: body.join('\n') });
      i = j + 1;
    } else { i += 1; }
  }
}
fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
console.log('extracted', out.length, 'diagrams');
