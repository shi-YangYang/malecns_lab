import type { Run } from './types';

export function drawRateChart(svg: SVGSVGElement, run: Run, group: string, frame: number, maxRate: number): void {
  const column = run.groups.indexOf(group);
  const width = 264, height = 132, left = 34, right = 254, top = 12, bottom = 108;
  const start = run.time_s[0] ?? 0;
  const end = run.time_s.at(-1) ?? 1;
  const span = Math.max(end - start, 1 / run.frame_hz);
  const x = (time: number): number => left + (time - start) / span * (right - left);
  const y = (rate: number): number => bottom - rate / maxRate * (bottom - top);
  let path = '', open = false;
  for (let i = 0; i < run.time_s.length; i++) {
    const rate = column >= 0 ? run.group_rates_hz[i]?.[column] : null;
    if (!run.valid[i] || rate === null || rate === undefined) { open = false; continue; }
    path += `${open ? 'H' : 'M'}${x(run.time_s[i]).toFixed(2)}${open ? 'V' : ','}${y(rate).toFixed(2)} `;
    open = true;
  }
  const cursor = x(run.time_s[frame] ?? start);
  const rate = run.valid[frame] && column >= 0 ? run.group_rates_hz[frame]?.[column] : null;
  const warmupEnd = Math.max(left, Math.min(right, x(run.warmup_s)));
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('aria-label', `${group} 群组平均放电率曲线，纵轴 Hz，横轴仿真秒；阶梯线连接原始记录帧。`);
  svg.innerHTML = `
    <rect x="${left}" y="${top}" width="${warmupEnd - left}" height="${bottom - top}" fill="#ffffff" opacity=".026" />
    ${[0, 0.5, 1].map(fraction => `<line x1="${left}" x2="${right}" y1="${y(maxRate * fraction)}" y2="${y(maxRate * fraction)}" stroke="#2b3637" stroke-dasharray="2 4"/><text x="${left - 7}" y="${y(maxRate * fraction) + 4}" text-anchor="end">${Math.round(maxRate * fraction)}</text>`).join('')}
    <path d="${path}" fill="none" stroke="#7ce0be" stroke-width="1.6" stroke-linejoin="round" />
    <line x1="${cursor}" x2="${cursor}" y1="${top}" y2="${bottom}" stroke="#cfb281" stroke-width="1" stroke-dasharray="3 3" />
    ${rate !== null && rate !== undefined ? `<circle cx="${cursor}" cy="${y(rate)}" r="3" fill="#b6efcf"/>` : ''}
    <text x="${left}" y="126">${start.toFixed(1)}</text>
    <text x="${right}" y="126" text-anchor="end">${end.toFixed(2)} 秒</text>`;
}
