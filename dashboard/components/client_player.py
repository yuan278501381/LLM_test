"""浏览器端感知器播放器：不触发 Streamlit 重跑，避免播放时布局闪烁。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_perceptron_payload(
    weight_trajectory: Sequence[np.ndarray],
    bias_trajectory: Sequence[np.ndarray],
    x: np.ndarray,
    mask_0: np.ndarray,
    mask_1: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> str:
    """序列化可信的数值轨迹，供同源 iframe 在浏览器内原位绘制。"""
    payload = {
        "weights": [
            [float(weight.ravel()[0]), float(weight.ravel()[1])] for weight in weight_trajectory
        ],
        "biases": [float(bias.ravel()[0]) for bias in bias_trajectory],
        "class0": x[mask_0].astype(float).tolist(),
        "class1": x[mask_1].astype(float).tolist(),
        "xRange": [float(x_range[0]), float(x_range[1])],
        "yRange": [float(y_range[0]), float(y_range[1])],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_timeline_controls(
    *,
    total_steps: int,
    event_name: str,
    title: str,
    badge: str,
    caption: str,
    progress_name: str = "时间演进",
    inspect_label: str = "当前状态",
    interval_ms: int = 420,
    initial_step: int | None = None,
) -> None:
    """渲染通用浏览器端时间轴；所有订阅图表都只在原位重绘。"""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    start_step = total_steps if initial_step is None else min(max(initial_step, 1), total_steps)
    event_json = json.dumps(event_name, ensure_ascii=False)
    title_json = json.dumps(title, ensure_ascii=False)
    badge_json = json.dumps(badge, ensure_ascii=False)
    caption_json = json.dumps(caption, ensure_ascii=False)
    progress_json = json.dumps(progress_name, ensure_ascii=False)
    inspect_json = json.dumps(inspect_label, ensure_ascii=False)
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}} body{{margin:0;padding:2px 0 0;background:transparent;color:#0f172a;
      font-family:'JetBrains Mono',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
    .status{{height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;
      border:1px solid #e2e8f0;border-radius:10px;background:#fff;box-shadow:0 2px 8px rgba(15,23,42,.03)}}
    .title{{display:flex;align-items:center;gap:9px;color:#1e40af;font-size:13px;font-weight:800;letter-spacing:0.02em}}
    .badge{{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:7px;background:#f5f3ff;
      border:1px solid #ddd6fe;color:#5b21b6;font-family:'JetBrains Mono',monospace;font-weight:800}}
    .state{{font:700 12px 'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;padding:5px 10px;border-radius:6px;
      color:#047857;background:#ecfdf5;border:1px solid #a7f3d0;font-variant-numeric:tabular-nums}}
    .buttons{{display:grid;grid-template-columns:1fr 1.1fr 1.8fr 1.1fr 1fr;gap:12px;margin-top:12px}}
    button{{height:42px;border-radius:8px;border:1px solid #cbd5e1;background:#fff;color:#0f172a;
      font-family:'JetBrains Mono',-apple-system,sans-serif;font-size:13px;font-weight:700;cursor:pointer;
      box-shadow:0 1px 3px rgba(15,23,42,0.04);transition:all .18s cubic-bezier(0.4,0,0.2,1)}}
    button:hover{{border-color:#2563eb;color:#1d4ed8;background:#f8fafc;box-shadow:0 3px 8px rgba(37,99,235,0.08);transform:translateY(-1px)}}
    button:active{{transform:translateY(1px) scale(0.99)}}
    #toggle{{background:#1d4ed8;border-color:#1d4ed8;color:#fff;box-shadow:0 2px 6px rgba(29,78,216,0.24)}}
    #toggle:hover{{background:#1e40af;border-color:#1e40af;color:#fff;box-shadow:0 4px 12px rgba(29,78,216,0.32)}}
    .progress-label{{margin-top:12px;font-size:12.5px;font-weight:600;color:#334155;font-variant-numeric:tabular-nums}}
    .track{{height:6px;margin-top:7px;border-radius:999px;background:#e2e8f0;overflow:hidden}}
    .fill{{height:100%;width:100%;background:linear-gradient(90deg,#1d4ed8 0%,#4338ca 100%);transform-origin:left center;transition:transform .38s cubic-bezier(0.16,1,0.3,1)}}
    .caption{{margin-top:10px;color:#64748b;font-size:12px;line-height:1.5}}
    @media(max-width:720px){{.buttons{{grid-template-columns:1fr 1fr 1.5fr 1fr 1fr;gap:5px}}button{{font-size:11px}}}}
    @media(prefers-reduced-motion:reduce){{*{{transition:none!important}}}}
    </style></head><body>
      <div class="status"><div class="title"><span id="badge" class="badge"></span>
        <span id="title"></span></div><span id="state" class="state"></span></div>
      <div class="buttons">
        <button id="first" title="回到序列起点">⏮ 起点</button>
        <button id="prev" title="暂停并回退一步">◀ 上一步</button>
        <button id="toggle" title="连续展示参数与图形变化">▶ 连续演播</button>
        <button id="next" title="暂停并前进一步">下一步 ▶</button>
        <button id="last" title="跳到最终收敛状态">终点 ⏭</button>
      </div>
      <div id="progress-label" class="progress-label"></div><div class="track"><div id="fill" class="fill"></div></div>
      <div id="caption" class="caption"></div>
    <script>
    (() => {{
      const total={total_steps}, eventName={event_json}; let step={start_step}; let playing=false; let timer=null;
      const state=document.getElementById('state'), toggle=document.getElementById('toggle');
      const label=document.getElementById('progress-label'), fill=document.getElementById('fill');
      document.getElementById('title').textContent={title_json};
      document.getElementById('badge').textContent={badge_json};
      document.getElementById('caption').textContent={caption_json};
      const progressName={progress_json}, inspectLabel={inspect_json};
      const stride=Math.max(1,Math.floor(total/50));
      function emit(){{
        const text=playing?'▶ 演播中':(step===total?'准备就绪':`Ⅱ 已暂停 · 可观察${{inspectLabel}}`);
        state.textContent=`${{text}} · STEP ${{step}}/${{total}}`;
        state.style.color=playing?'#1d4ed8':(step===total?'#047857':'#92400e');
        state.style.background=playing?'#eff6ff':(step===total?'#ecfdf5':'#fffbeb');
        state.style.borderColor=playing?'#bfdbfe':(step===total?'#a7f3d0':'#fde68a');
        toggle.textContent=playing?'Ⅱ 暂停观察':'▶ 连续演播';
        label.textContent=`${{progressName}} · Step ${{step}}/${{total}}`;
        fill.style.transform=`scaleX(${{step/total}})`;
        window.parent.dispatchEvent(new CustomEvent(eventName,{{detail:{{step,total}}}}));
      }}
      function stop(){{playing=false;if(timer)clearInterval(timer);timer=null;emit()}}
      function start(){{if(step>=total)step=1;playing=true;emit();timer=setInterval(()=>{{
        step=Math.min(total,step+stride);if(step>=total){{stop();return}}emit();
      }},{max(120, interval_ms)})}}
      document.getElementById('first').onclick=()=>{{stop();step=1;emit()}};
      document.getElementById('prev').onclick=()=>{{stop();step=Math.max(1,step-1);emit()}};
      toggle.onclick=()=>playing?stop():start();
      document.getElementById('next').onclick=()=>{{stop();step=Math.min(total,step+1);emit()}};
      document.getElementById('last').onclick=()=>{{stop();step=total;emit()}};
      document.addEventListener('keydown',e=>{{if(e.code==='Space'){{e.preventDefault();toggle.click()}}
        if(e.code==='ArrowLeft')document.getElementById('prev').click();
        if(e.code==='ArrowRight')document.getElementById('next').click();}});
      emit();
    }})();
    </script></body></html>
    """
    st.iframe(html, height=192, width="stretch", tab_index=0)


def render_player_controls(payload_json: str) -> None:
    """渲染感知器播放器；保留稳定入口供页面与测试调用。"""
    total_steps = len(json.loads(payload_json)["weights"])
    render_timeline_controls(
        total_steps=total_steps,
        event_name="nn:m1-step",
        title="[TIME-TRAVEL PLAYER // 训练时空演播厅]",
        badge="D",
        caption="浏览器原位动画 · 可随时暂停，并用上一步/下一步精确检查任意参数状态。",
        progress_name="训练演进",
        inspect_label="当前参数",
    )


def _canvas_base_css(height: int) -> str:
    return f"""
    *{{box-sizing:border-box}}body{{margin:0;background:transparent;color:#0f172a;
      font-family:'JetBrains Mono',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;overflow:hidden}}
    .equation{{height:32px;line-height:30px;padding:0 12px;border:1px solid #e2e8f0;border-radius:7px;
      background:#f8fafc;color:#1e40af;font:700 12px/30px 'JetBrains Mono',ui-monospace,SFMono-Regular,Consolas,monospace;
      font-variant-numeric:tabular-nums;margin-bottom:6px;box-shadow:0 1px 3px rgba(15,23,42,0.02)}}
    canvas{{display:block;width:100%;height:{height}px;background:#fff}}
    """


def render_boundary_canvas(payload_json: str) -> None:
    """在固定 Canvas 中原位更新概率背景、决策线与样本点。"""
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>{_canvas_base_css(398)}</style></head><body>
    <div id="equation" class="equation"></div><canvas id="plot" aria-label="动态决策边界图"></canvas>
    <script>
    (()=>{{const data={payload_json};const canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
      const eq=document.getElementById('equation');let step=data.weights.length;let W=0,H=398,dpr=1;
      const m={{l:58,r:16,t:12,b:48}};
      function sx(x){{return m.l+(x-data.xRange[0])/(data.xRange[1]-data.xRange[0])*(W-m.l-m.r)}}
      function sy(y){{return H-m.b-(y-data.yRange[0])/(data.yRange[1]-data.yRange[0])*(H-m.t-m.b)}}
      function resize(){{W=Math.max(420,canvas.clientWidth);dpr=window.devicePixelRatio||1;canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw()}}
      function grid(){{ctx.strokeStyle='#e8edf4';ctx.lineWidth=1;ctx.font='11px ui-monospace,monospace';ctx.fillStyle='#64748b';
        for(let i=0;i<=4;i++){{let x=data.xRange[0]+(data.xRange[1]-data.xRange[0])*i/4,px=sx(x);ctx.beginPath();ctx.moveTo(px,m.t);ctx.lineTo(px,H-m.b);ctx.stroke();ctx.textAlign='center';ctx.fillText(x.toFixed(1),px,H-m.b+19);
          let y=data.yRange[0]+(data.yRange[1]-data.yRange[0])*i/4,py=sy(y);ctx.beginPath();ctx.moveTo(m.l,py);ctx.lineTo(W-m.r,py);ctx.stroke();ctx.textAlign='right';ctx.fillText(y.toFixed(1),m.l-8,py+4)}}
        ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText('Feature x₁',(m.l+W-m.r)/2,H-10);ctx.save();ctx.translate(15,(m.t+H-m.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText('Feature x₂',0,0);ctx.restore()}}
      function background(w,b){{const ow=120,oh=72,off=document.createElement('canvas');off.width=ow;off.height=oh;const oc=off.getContext('2d'),im=oc.createImageData(ow,oh);
        for(let j=0;j<oh;j++)for(let i=0;i<ow;i++){{let x=data.xRange[0]+(data.xRange[1]-data.xRange[0])*i/(ow-1),y=data.yRange[1]-(data.yRange[1]-data.yRange[0])*j/(oh-1),p=1/(1+Math.exp(-(w[0]*x+w[1]*y+b))),k=(j*ow+i)*4;
          im.data[k]=Math.round(98+113*(1-p));im.data[k+1]=Math.round(105+105*(1-p));im.data[k+2]=Math.round(153+90*p);im.data[k+3]=112}}oc.putImageData(im,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(off,m.l,m.t,W-m.l-m.r,H-m.t-m.b)}}
      function points(rows,color){{ctx.fillStyle=color;ctx.strokeStyle='#fff';ctx.lineWidth=1.5;for(const p of rows){{ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),4.2,0,Math.PI*2);ctx.fill();ctx.stroke()}}}}
      function line(w,b){{let pts=[],xmin=data.xRange[0],xmax=data.xRange[1],ymin=data.yRange[0],ymax=data.yRange[1];
        if(Math.abs(w[1])>1e-9){{for(const x of [xmin,xmax]){{let y=-(w[0]*x+b)/w[1];if(y>=ymin&&y<=ymax)pts.push([x,y])}}}}
        if(Math.abs(w[0])>1e-9){{for(const y of [ymin,ymax]){{let x=-(w[1]*y+b)/w[0];if(x>=xmin&&x<=xmax&&!pts.some(p=>Math.abs(p[0]-x)<1e-6))pts.push([x,y])}}}}
        if(pts.length>=2){{ctx.strokeStyle='#111827';ctx.lineWidth=3.2;ctx.beginPath();ctx.moveTo(sx(pts[0][0]),sy(pts[0][1]));ctx.lineTo(sx(pts[1][0]),sy(pts[1][1]));ctx.stroke()}}}}
      function draw(){{if(!W)return;let w=data.weights[step-1],b=data.biases[step-1];ctx.clearRect(0,0,W,H);background(w,b);grid();line(w,b);points(data.class0,'#1d4ed8');points(data.class1,'#be123c');
        let sign=b>=0?'+':'−';eq.textContent=`分界实线方程：${{w[0].toFixed(2)}}x₁ + ${{w[1].toFixed(2)}}x₂ ${{sign}} ${{Math.abs(b).toFixed(2)}} = 0`}}
      window.parent.addEventListener('nn:m1-step',e=>{{step=e.detail.step;requestAnimationFrame(draw)}});new ResizeObserver(resize).observe(canvas);resize();
    }})();</script></body></html>"""
    st.iframe(html, height=432, width="stretch")


def render_trajectory_canvas(payload_json: str) -> None:
    """在固定 Canvas 中原位更新权重寻优轨迹。"""
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>{_canvas_base_css(370)}</style></head><body>
    <canvas id="plot" aria-label="动态权重寻优轨迹图"></canvas><script>
    (()=>{{const data={payload_json},canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');let step=data.weights.length,W=0,H=370,dpr=1;
      const m={{l:58,r:24,t:16,b:48}},all=data.weights,xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
      const pad=(a,b)=>Math.max((b-a)*.13,.08),px=pad(Math.min(...xs),Math.max(...xs)),py=pad(Math.min(...ys),Math.max(...ys));
      const xr=[Math.min(...xs)-px,Math.max(...xs)+px],yr=[Math.min(...ys)-py,Math.max(...ys)+py];
      function sx(x){{return m.l+(x-xr[0])/(xr[1]-xr[0])*(W-m.l-m.r)}}function sy(y){{return H-m.b-(y-yr[0])/(yr[1]-yr[0])*(H-m.t-m.b)}}
      function resize(){{W=Math.max(420,canvas.clientWidth);dpr=window.devicePixelRatio||1;canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw()}}
      function draw(){{if(!W)return;ctx.clearRect(0,0,W,H);ctx.font='11px ui-monospace,monospace';
        for(let i=0;i<=4;i++){{ctx.strokeStyle='#e8edf4';let x=xr[0]+(xr[1]-xr[0])*i/4,xx=sx(x),y=yr[0]+(yr[1]-yr[0])*i/4,yy=sy(y);ctx.beginPath();ctx.moveTo(xx,m.t);ctx.lineTo(xx,H-m.b);ctx.stroke();ctx.beginPath();ctx.moveTo(m.l,yy);ctx.lineTo(W-m.r,yy);ctx.stroke();ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText(x.toFixed(2),xx,H-m.b+19);ctx.textAlign='right';ctx.fillText(y.toFixed(2),m.l-8,yy+4)}}
        let pts=all.slice(0,step);ctx.strokeStyle='#1d4ed8';ctx.lineWidth=2.8;ctx.lineJoin='round';ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(sx(p[0]),sy(p[1])):ctx.moveTo(sx(p[0]),sy(p[1])));ctx.stroke();
        for(let i=0;i<pts.length;i+=Math.max(1,Math.floor(pts.length/24))){{ctx.fillStyle='#93c5fd';ctx.beginPath();ctx.arc(sx(pts[i][0]),sy(pts[i][1]),3,0,Math.PI*2);ctx.fill()}}
        let first=all[0],cur=pts[pts.length-1];ctx.fillStyle='#be123c';ctx.beginPath();ctx.arc(sx(first[0]),sy(first[1]),6,0,Math.PI*2);ctx.fill();ctx.fillStyle='#047857';ctx.save();ctx.translate(sx(cur[0]),sy(cur[1]));ctx.rotate(Math.PI/4);ctx.fillRect(-6,-6,12,12);ctx.restore();
        ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText('Parameter w₁',(m.l+W-m.r)/2,H-10);ctx.save();ctx.translate(15,(m.t+H-m.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText('Parameter w₂',0,0);ctx.restore();ctx.textAlign='right';ctx.fillText(`Step ${{step}}/${{all.length}}`,W-m.r,m.t+4)}}
      window.parent.addEventListener('nn:m1-step',e=>{{step=e.detail.step;requestAnimationFrame(draw)}});new ResizeObserver(resize).observe(canvas);resize();
    }})();</script></body></html>"""
    st.iframe(html, height=374, width="stretch")


def build_video_payload(frames: np.ndarray, frame_diffs: Sequence[float]) -> str:
    """序列化视频帧及相邻帧运动能量，供浏览器端时间轴使用。"""
    frame_array = np.asarray(frames, dtype=float)
    if frame_array.ndim == 4 and frame_array.shape[1] == 1:
        frame_array = frame_array[:, 0]
    if frame_array.ndim != 3 or len(frame_array) < 1:
        raise ValueError("frames must have shape (T, H, W) or (T, 1, H, W)")
    diffs = [0.0, *[float(value) for value in frame_diffs]]
    if len(diffs) < len(frame_array):
        diffs.extend([diffs[-1]] * (len(frame_array) - len(diffs)))
    payload = {
        "frames": frame_array.tolist(),
        "energy": diffs[: len(frame_array)],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_video_timeline(payload_json: str) -> None:
    """在固定 Canvas 中平滑播放视频帧，并同步移动运动能量游标。"""
    html = f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
    *{{box-sizing:border-box}}body{{margin:0;background:transparent;color:#0f172a;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;overflow:hidden}}
    .shell{{height:360px;border:1px solid #e2e8f0;border-radius:10px;background:#fff;overflow:hidden}}
    canvas{{display:block;width:100%;height:100%}}
    @media(prefers-reduced-motion:reduce){{canvas{{scroll-behavior:auto}}}}
    </style></head><body><div class="shell"><canvas id="plot"
      aria-label="视频帧与运动能量同步播放图"></canvas></div><script>
    (()=>{{const data={payload_json},canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
      const total=data.frames.length,H=360;let W=0,dpr=1,current=0,from=0,target=0,started=0,raf=0;
      const duration=300,energyMax=Math.max(...data.energy,1e-6)*1.12;
      function viridis(v){{v=Math.max(0,Math.min(1,v));const stops=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
        const p=v*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),t=p-i,a=stops[i],b=stops[i+1];
        return [0,1,2].map(k=>Math.round(a[k]+(b[k]-a[k])*t))}}
      function resize(){{W=Math.max(640,canvas.clientWidth);dpr=window.devicePixelRatio||1;canvas.width=W*dpr;canvas.height=H*dpr;
        ctx.setTransform(dpr,0,0,dpr,0,0);draw(performance.now())}}
      function frameImage(index,nextIndex,mix,x,y,size){{const a=data.frames[index],b=data.frames[nextIndex],rows=a.length,cols=a[0].length;
        const off=document.createElement('canvas');off.width=cols;off.height=rows;const oc=off.getContext('2d'),im=oc.createImageData(cols,rows);
        for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){{const v=a[r][c]*(1-mix)+b[r][c]*mix,col=viridis(v),k=(r*cols+c)*4;
          im.data[k]=col[0];im.data[k+1]=col[1];im.data[k+2]=col[2];im.data[k+3]=255}}oc.putImageData(im,0,0);
        ctx.imageSmoothingEnabled=true;ctx.drawImage(off,x,y,size,size)}}
      function draw(now){{if(!W)return;const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
        const mix=reduce||from===target?1:Math.min(1,(now-started)/duration),visual=from+(target-from)*mix;
        ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
        const gap=44,leftW=Math.min(W*.42,400),imageSize=Math.min(285,leftW-54),ix=(leftW-imageSize)/2,iy=45;
        frameImage(from,target,mix,ix,iy,imageSize);ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1;ctx.strokeRect(ix-.5,iy-.5,imageSize+1,imageSize+1);
        ctx.fillStyle='#0f172a';ctx.font='700 17px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(`当前物理帧 · T=${{Math.round(visual)}}`,leftW/2,25);
        ctx.fillStyle='#64748b';ctx.font='12px ui-monospace,monospace';ctx.fillText('原位插值过渡 · 无页面重绘',leftW/2,H-12);
        const x0=leftW+gap,x1=W-34,y0=54,y1=H-56,cw=x1-x0,ch=y1-y0;
        ctx.fillStyle='#0f172a';ctx.font='700 17px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText('运动能量 (Frame Difference MSE)',(x0+x1)/2,25);
        ctx.font='11px ui-monospace,monospace';for(let i=0;i<=4;i++){{const y=y1-ch*i/4;ctx.strokeStyle='#e8edf4';ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke();ctx.fillStyle='#64748b';ctx.textAlign='right';ctx.fillText((energyMax*i/4).toFixed(3),x0-8,y+4)}}
        const sx=i=>x0+(total===1?0:i/(total-1))*cw,sy=e=>y1-e/energyMax*ch;
        ctx.strokeStyle='#be123c';ctx.lineWidth=2.5;ctx.lineJoin='round';ctx.beginPath();data.energy.forEach((e,i)=>i?ctx.lineTo(sx(i),sy(e)):ctx.moveTo(sx(i),sy(e)));ctx.stroke();
        const e=data.energy[from]*(1-mix)+data.energy[target]*mix,px=sx(visual),py=sy(e);ctx.fillStyle='#be123c';ctx.beginPath();ctx.arc(px,py,6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
        ctx.fillStyle='#4338ca';ctx.font='700 12px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(`T=${{Math.round(visual)}} · MSE=${{e.toFixed(4)}}`,px,Math.max(y0+15,py-13));
        ctx.fillStyle='#64748b';ctx.font='11px ui-monospace,monospace';ctx.fillText('视频时间步 (Frame Index)',(x0+x1)/2,H-17);
        if(mix<1)raf=requestAnimationFrame(draw);else{{current=target;from=target}}}}
      window.parent.addEventListener('nn:m12-frame',e=>{{const next=Math.min(total-1,Math.max(0,e.detail.step-1));
        from=current;target=next;started=performance.now();cancelAnimationFrame(raf);raf=requestAnimationFrame(draw)}});
      new ResizeObserver(resize).observe(canvas);resize();
    }})();</script></body></html>"""
    st.iframe(html, height=364, width="stretch")
