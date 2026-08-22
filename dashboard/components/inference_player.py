"""浏览器端多层网络前向传播教学动画。"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import streamlit as st


class TrainingTrajectoryRecorder:
    """按固定检查点记录真实模型状态，避免用视觉淡入伪装训练过程。"""

    def __init__(
        self,
        x: np.ndarray,
        probe: tuple[float, float],
        total_epochs: int,
        resolution: int = 52,
        max_frames: int = 42,
    ) -> None:
        margin = 0.3
        self.x_range = (float(x[:, 0].min() - margin), float(x[:, 0].max() + margin))
        self.y_range = (float(x[:, 1].min() - margin), float(x[:, 1].max() + margin))
        gx = np.linspace(*self.x_range, resolution)
        gy = np.linspace(*self.y_range, resolution)
        xx, yy = np.meshgrid(gx, gy)
        self.grid = np.c_[xx.ravel(), yy.ravel()]
        self.resolution = resolution
        self.probe = np.asarray([probe], dtype=float)
        self.total_epochs = max(1, int(total_epochs))
        self.interval = max(1, math.ceil(self.total_epochs / max_frames))
        self.frames: list[dict[str, Any]] = []

    def _capture(self, epoch: int, model: Any, loss: float | None, accuracy: float | None) -> None:
        signal = self.probe
        activations: list[list[float]] = [signal.ravel().astype(float).tolist()]
        weights: list[list[list[float]]] = []
        for layer in model.layers:
            signal = (
                layer.forward(signal, training=False)
                if hasattr(layer, "weights")
                else layer.forward(signal)
            )
            if hasattr(layer, "weights"):
                weights.append(np.asarray(layer.weights, dtype=float).tolist())
            else:
                activations.append(np.asarray(signal, dtype=float).ravel().tolist())
        field = model.predict(self.grid).reshape(self.resolution, self.resolution)
        self.frames.append(
            {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "weights": weights,
                "activations": activations,
                "field": np.asarray(field, dtype=float).tolist(),
                "probeProb": float(signal.ravel()[0]),
            }
        )

    def capture_initial(self, model: Any) -> None:
        self._capture(0, model, None, None)

    def on_epoch_end(self, epoch: int, model: Any, logs: dict[str, float]) -> None:
        epoch_number = epoch + 1
        if epoch_number % self.interval == 0 or epoch_number == self.total_epochs:
            self._capture(epoch_number, model, float(logs["loss"]), float(logs["accuracy"]))


def build_training_payload(
    recorder: TrainingTrajectoryRecorder,
    x: np.ndarray,
    y: np.ndarray,
    probe: tuple[float, float],
) -> str:
    """序列化真实训练检查点，供拓扑与决策场同步播放。"""
    payload = {
        "frames": recorder.frames,
        "points": np.asarray(x, dtype=float).tolist(),
        "labels": np.asarray(y).ravel().astype(int).tolist(),
        "probe": [float(probe[0]), float(probe[1])],
        "xRange": list(recorder.x_range),
        "yRange": list(recorder.y_range),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _base_css(height: int) -> str:
    return f"""
    *{{box-sizing:border-box}}body{{margin:0;background:transparent;color:#0f172a;
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;overflow:hidden}}
    .shell{{height:{height}px;border:1px solid #e2e8f0;border-radius:9px;background:#fff;overflow:hidden}}
    canvas{{display:block;width:100%;height:100%}}
    """


def render_network_signal_canvas(payload_json: str) -> None:
    """同步插值真实检查点中的权重与探针神经元响应。"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{_base_css(420)}</style></head>
    <body><div class="shell"><canvas id="plot" aria-label="多层网络逐层前向传播动画"></canvas></div><script>
    (()=>{{const data={payload_json},canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
      const frames=data.frames,sizes=frames[0].activations.map(a=>a.length),n=sizes.length,H=420,limit=10;
      let W=0,dpr=1,current=0,from=0,target=0,started=0,raf=0;const duration=360;
      const labels=['INPUT',...sizes.slice(1,-1).map((_,i)=>`HIDDEN #${{i+1}}`),'OUTPUT'];
      function positions(){{return sizes.map((count,l)=>{{const shown=Math.min(count,limit),x=58+l*(W-116)/(n-1),gap=Math.min(43,270/Math.max(shown-1,1));
        const top=190-gap*(shown-1)/2;return Array.from({{length:shown}},(_,i)=>[x,top+i*gap])}})}}
      function color(a,alpha=1){{if(a>.5)return `rgba(37,99,235,${{alpha}})`;if(a>.05)return `rgba(5,150,105,${{alpha}})`;
        if(a<-.05)return `rgba(225,29,72,${{alpha}})`;return `rgba(226,232,240,${{alpha}})`}}
      function resize(){{W=Math.max(520,canvas.clientWidth);dpr=devicePixelRatio||1;canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw(performance.now())}}
      function lerp(a,b,t){{return a+(b-a)*t}}function draw(now){{if(!W)return;const pos=positions(),t=from===target?1:Math.min(1,(now-started)/duration),fa=frames[from],fb=frames[target];ctx.clearRect(0,0,W,H);
        ctx.fillStyle='#0f172a';ctx.font='700 14px ui-monospace,monospace';ctx.textAlign='left';ctx.fillText('TRAINING DYNAMICS // 权重与探针响应同步演化',18,25);
        for(let l=0;l<n-1;l++)for(let i=0;i<pos[l].length;i++)for(let j=0;j<pos[l+1].length;j++){{
          const w=lerp(fa.weights[l]?.[i]?.[j]??0,fb.weights[l]?.[i]?.[j]??0,t),alpha=Math.min(.78,.12+Math.abs(w)*.36);ctx.strokeStyle=w>=0?`rgba(37,99,235,${{alpha}})`:`rgba(225,29,72,${{alpha}})`;
          ctx.lineWidth=Math.min(4.2,.6+Math.abs(w)*2.1);ctx.beginPath();ctx.moveTo(...pos[l][i]);ctx.lineTo(...pos[l+1][j]);ctx.stroke();
          if(t<1&&Math.abs(w)>.22){{const x=lerp(pos[l][i][0],pos[l+1][j][0],t),y=lerp(pos[l][i][1],pos[l+1][j][1],t);ctx.fillStyle=w>=0?'#60a5fa':'#fb7185';ctx.beginPath();ctx.arc(x,y,2.8,0,Math.PI*2);ctx.fill()}}}}
        for(let l=0;l<n;l++){{for(let i=0;i<pos[l].length;i++){{const a=lerp(fa.activations[l][i]??0,fb.activations[l][i]??0,t),x=pos[l][i][0],y=pos[l][i][1];
          ctx.fillStyle='rgba(99,102,241,.08)';ctx.beginPath();ctx.arc(x,y,15+Math.min(6,Math.abs(a)*4),0,Math.PI*2);ctx.fill();ctx.fillStyle=color(a);ctx.strokeStyle='#cbd5e1';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,12,0,Math.PI*2);ctx.fill();ctx.stroke();
          ctx.fillStyle=Math.abs(a)>.05?'#fff':'#64748b';ctx.font='9px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(String(i+1),x,y+3)}}
          ctx.fillStyle='#0f172a';ctx.font='700 11px ui-monospace,monospace';ctx.fillText(labels[l],pos[l][0][0],365);ctx.fillStyle='#64748b';ctx.font='10px ui-monospace,monospace';ctx.fillText(`dim=${{sizes[l]}}`,pos[l][0][0],382)}}
        const epoch=Math.round(lerp(fa.epoch,fb.epoch,t)),loss=fb.loss==null?'初始化':`Loss ${{fb.loss.toFixed(4)}} · Acc ${{(fb.accuracy*100).toFixed(1)}}%`;ctx.fillStyle='#4338ca';ctx.font='700 12px ui-monospace,monospace';ctx.textAlign='left';ctx.fillText(`真实训练检查点 · Epoch ${{epoch}} · ${{loss}}`,18,408);
        if(t<1)raf=requestAnimationFrame(draw);else{{current=target;from=target}}}}
      window.parent.addEventListener('nn:m2-train',e=>{{const next=Math.min(frames.length-1,Math.max(0,e.detail.step-1));from=current;target=next;started=performance.now();cancelAnimationFrame(raf);raf=requestAnimationFrame(draw)}});
      new ResizeObserver(resize).observe(canvas);resize();
    }})();</script></body></html>"""
    st.iframe(html, height=424, width="stretch")


def render_probe_manifold_canvas(payload_json: str) -> None:
    """同步插值真实训练检查点中的概率场、边界和探针预测。"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>{_base_css(420)}</style></head>
    <body><div class="shell"><canvas id="plot" aria-label="决策流形真实训练演化动画"></canvas></div><script>
    (()=>{{const data={payload_json},frames=data.frames,canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
      const H=420,m={{l:52,r:18,t:52,b:48}};let W=0,dpr=1,current=0,from=0,target=0,started=0,raf=0;const duration=360;
      const sx=x=>m.l+(x-data.xRange[0])/(data.xRange[1]-data.xRange[0])*(W-m.l-m.r),sy=y=>H-m.b-(y-data.yRange[0])/(data.yRange[1]-data.yRange[0])*(H-m.t-m.b);
      function resize(){{W=Math.max(480,canvas.clientWidth);dpr=devicePixelRatio||1;canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);draw(performance.now())}}
      function field(a,b,t){{const rows=a.length,cols=a[0].length,pw=W-m.l-m.r,ph=H-m.t-m.b,cw=pw/(cols-1),ch=ph/(rows-1),val=(r,c)=>a[r][c]+(b[r][c]-a[r][c])*t;
        const off=document.createElement('canvas');off.width=cols;off.height=rows;const oc=off.getContext('2d'),im=oc.createImageData(cols,rows);
        for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){{const p=val(r,c),strength=.14+.34*Math.abs(p-.5)*2,base=p<.5?[79,99,223]:[184,50,75],k=((rows-1-r)*cols+c)*4;
          im.data[k]=Math.round(255+(base[0]-255)*strength);im.data[k+1]=Math.round(255+(base[1]-255)*strength);im.data[k+2]=Math.round(255+(base[2]-255)*strength);im.data[k+3]=255}}oc.putImageData(im,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(off,m.l,m.t,pw,ph);
        function edge(p1,p2,v1,v2){{if((v1<.5)===(v2<.5))return null;const q=(.5-v1)/(v2-v1);return [p1[0]+(p2[0]-p1[0])*q,p1[1]+(p2[1]-p1[1])*q]}}
        ctx.strokeStyle='#111827';ctx.lineWidth=3;ctx.lineCap='round';for(let r=0;r<rows-1;r++)for(let c=0;c<cols-1;c++){{const x=m.l+c*cw,yb=H-m.b-r*ch,yt=yb-ch,v0=val(r,c),v1=val(r,c+1),v2=val(r+1,c+1),v3=val(r+1,c),pts=[];
          for(const p of [edge([x,yb],[x+cw,yb],v0,v1),edge([x+cw,yb],[x+cw,yt],v1,v2),edge([x+cw,yt],[x,yt],v2,v3),edge([x,yt],[x,yb],v3,v0)])if(p)pts.push(p);
          if(pts.length>=2){{ctx.beginPath();ctx.moveTo(...pts[0]);ctx.lineTo(...pts[1]);ctx.stroke()}}if(pts.length===4){{ctx.beginPath();ctx.moveTo(...pts[2]);ctx.lineTo(...pts[3]);ctx.stroke()}}}}}}
      function draw(now){{if(!W)return;const t=from===target?1:Math.min(1,(now-started)/duration),fa=frames[from],fb=frames[target];ctx.clearRect(0,0,W,H);field(fa.field,fb.field,t);
        ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;for(let i=0;i<=4;i++){{const x=m.l+(W-m.l-m.r)*i/4,y=m.t+(H-m.t-m.b)*i/4;ctx.beginPath();ctx.moveTo(x,m.t);ctx.lineTo(x,H-m.b);ctx.stroke();ctx.beginPath();ctx.moveTo(m.l,y);ctx.lineTo(W-m.r,y);ctx.stroke()}}
        data.points.forEach((p,i)=>{{ctx.fillStyle=data.labels[i]?'#b8324b':'#4f63df';ctx.strokeStyle='#fff';ctx.lineWidth=1.4;ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),3.5,0,Math.PI*2);ctx.fill();ctx.stroke()}});
        const px=sx(data.probe[0]),py=sy(data.probe[1]);ctx.strokeStyle='#b45309';ctx.lineWidth=3;ctx.beginPath();ctx.arc(px,py,12,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(px-9,py);ctx.lineTo(px+9,py);ctx.moveTo(px,py-9);ctx.lineTo(px,py+9);ctx.stroke();
        const prob=fa.probeProb+(fb.probeProb-fa.probeProb)*t,epoch=Math.round(fa.epoch+(fb.epoch-fa.epoch)*t);ctx.fillStyle='#92400e';ctx.font='700 10px ui-monospace,monospace';ctx.fillText('PROBE',px+13,py-13);ctx.fillStyle='#0f172a';ctx.font='700 13px ui-monospace,monospace';ctx.textAlign='left';ctx.fillText('DECISION FIELD // 真实训练边界演化',14,18);
        ctx.fillStyle='#4338ca';ctx.font='700 11px ui-monospace,monospace';ctx.textAlign='right';ctx.fillText(`Epoch ${{epoch}} · Probe ${{(prob*100).toFixed(1)}}% · Class ${{prob>=.5?1:0}}`,W-18,39);ctx.textAlign='center';ctx.fillStyle='#64748b';ctx.font='11px ui-monospace,monospace';ctx.fillText('Feature x₁',(m.l+W-m.r)/2,H-12);
        if(t<1)raf=requestAnimationFrame(draw);else{{current=target;from=target}}}}
      window.parent.addEventListener('nn:m2-train',e=>{{const next=Math.min(frames.length-1,Math.max(0,e.detail.step-1));from=current;target=next;started=performance.now();cancelAnimationFrame(raf);raf=requestAnimationFrame(draw)}});
      new ResizeObserver(resize).observe(canvas);resize();
    }})();</script></body></html>"""
    st.iframe(html, height=424, width="stretch")
