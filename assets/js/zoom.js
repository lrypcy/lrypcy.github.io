/* zoom.js — 图片 / Mermaid 图表全屏查看器
 * 能力：点击 ⛶ 或图本身全屏；滚轮 / 双指捏合缩放；拖拽平移；双击切换；Esc 关闭
 * 无依赖。由 _layouts/default.html 引入（defer）。
 * 对外暴露 window.PCYZoom.attach(root)：mermaid 异步渲染完成后可再调用一次补挂。
 */
(function () {
  'use strict';

  var MIN_SCALE = 0.08;
  var MAX_SCALE = 12;
  var FIT_MAX = 3;      // 打开时"适应屏幕"的最大放大倍数
  var PADDING = 56;     // 适应屏幕时四周留白

  var overlay, bar, stage, content, scaleEl;
  var isOpen = false;
  var scale = 1, tx = 0, ty = 0;
  var baseW = 1, baseH = 1, fitScale = 1;
  var userZoomed = false;

  var pointers = {};
  var pointerCount = 0;
  var dragging = false;
  var lastPt = null;
  var pinchDist = 0;
  var animTimer = null;
  var lastFocus = null;
  var prevBodyPad = '';

  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
  function $(sel, root) { return (root || document).querySelector(sel); }

  /* ---------- 构建 DOM ---------- */
  function build() {
    if (document.getElementById('pcy-lightbox')) return;
    overlay = document.createElement('div');
    overlay.id = 'pcy-lightbox';
    overlay.className = 'pcy-lb';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', '图片查看器');
    overlay.innerHTML =
      '<div class="pcy-lb-bar">' +
        '<button type="button" class="pcy-lb-btn" data-act="out" title="缩小 (-)" aria-label="缩小">−</button>' +
        '<span class="pcy-lb-scale" id="pcyLbScale">100%</span>' +
        '<button type="button" class="pcy-lb-btn" data-act="in" title="放大 (+)" aria-label="放大">＋</button>' +
        '<button type="button" class="pcy-lb-btn pcy-lb-wide" data-act="fit" title="适应屏幕 (0)">适应屏幕</button>' +
        '<button type="button" class="pcy-lb-btn" data-act="one" title="原始大小 (1)">1:1</button>' +
        '<span class="pcy-lb-spacer"></span>' +
        '<button type="button" class="pcy-lb-btn pcy-lb-close" data-act="close" title="关闭 (Esc)" aria-label="关闭">✕</button>' +
      '</div>' +
      '<div class="pcy-lb-stage" id="pcyLbStage">' +
        '<div class="pcy-lb-content" id="pcyLbContent"></div>' +
      '</div>' +
      '<div class="pcy-lb-hint">滚轮 / 双指缩放 · 拖拽平移 · 双击放大 · <kbd>+</kbd> <kbd>−</kbd> <kbd>0</kbd> 适配 · <kbd>Esc</kbd> 关闭</div>';

    document.body.appendChild(overlay);
    bar = $('.pcy-lb-bar', overlay);
    stage = $('#pcyLbStage', overlay);
    content = $('#pcyLbContent', overlay);
    scaleEl = $('#pcyLbScale', overlay);

    bar.addEventListener('click', function (e) {
      var btn = e.target.closest('.pcy-lb-btn');
      if (!btn) return;
      var act = btn.getAttribute('data-act');
      if (act === 'close') close();
      else if (act === 'in') { animate(); zoomAt(null, null, 1.25); }
      else if (act === 'out') { animate(); zoomAt(null, null, 1 / 1.25); }
      else if (act === 'fit') { animate(); fit(); }
      else if (act === 'one') { animate(); setScale(1); }
    });

    // 滚轮缩放（以指针为锚点）
    stage.addEventListener('wheel', function (e) {
      if (!isOpen) return;
      e.preventDefault();
      var d = e.deltaY;
      if (e.deltaMode === 1) d *= 16;        // 行
      else if (e.deltaMode === 2) d *= 100;  // 页
      zoomAt(e.clientX, e.clientY, Math.exp(-clamp(d, -240, 240) * 0.0022));
    }, { passive: false });

    // 指针：单指拖拽平移 / 双指捏合缩放
    stage.addEventListener('pointerdown', function (e) {
      if (!isOpen) return;
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      pointers[e.pointerId] = { x: e.clientX, y: e.clientY };
      pointerCount++;
      if (pointerCount === 1) {
        dragging = true;
        lastPt = { x: e.clientX, y: e.clientY };
        stage.classList.add('dragging');
      } else if (pointerCount === 2) {
        dragging = false;
        pinchDist = twoPointerDist();
      }
      try { stage.setPointerCapture(e.pointerId); } catch (err) {}
    });

    stage.addEventListener('pointermove', function (e) {
      if (!isOpen || !pointers[e.pointerId]) return;
      pointers[e.pointerId] = { x: e.clientX, y: e.clientY };
      if (pointerCount >= 2) {
        var d = twoPointerDist();
        if (pinchDist > 0 && d > 0) {
          var mid = twoPointerMid();
          zoomAt(mid.x, mid.y, d / pinchDist);
        }
        pinchDist = d;
      } else if (dragging && lastPt) {
        tx += e.clientX - lastPt.x;
        ty += e.clientY - lastPt.y;
        lastPt = { x: e.clientX, y: e.clientY };
        apply();
      }
    });

    function endPointer(e) {
      if (pointers[e.pointerId]) {
        delete pointers[e.pointerId];
        pointerCount = Math.max(0, pointerCount - 1);
      }
      if (pointerCount === 0) {
        dragging = false;
        lastPt = null;
        stage.classList.remove('dragging');
      } else if (pointerCount === 1) {
        var id = Object.keys(pointers)[0];
        lastPt = { x: pointers[id].x, y: pointers[id].y };
        dragging = true;
        pinchDist = 0;
      }
      try { stage.releasePointerCapture(e.pointerId); } catch (err) {}
    }
    stage.addEventListener('pointerup', endPointer);
    stage.addEventListener('pointercancel', endPointer);

    // 双击：适应屏幕 ⇄ 放大
    stage.addEventListener('dblclick', function (e) {
      if (!isOpen) return;
      animate();
      if (Math.abs(scale - fitScale) < 0.02) zoomAt(e.clientX, e.clientY, 2);
      else fit();
    });

    // 点击空白处关闭
    stage.addEventListener('click', function (e) {
      if (!isOpen || e.target.closest('.pcy-lb-content')) return;
      if (Math.abs(scale - fitScale) < 0.02) close();
      else { animate(); fit(); }
    });

    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', function () {
      if (!isOpen) return;
      measureFit();
      if (!userZoomed) { tx = 0; ty = 0; scale = fitScale; apply(); }
    });
  }

  function twoPointerIds() { return Object.keys(pointers).slice(0, 2); }
  function twoPointerDist() {
    var ids = twoPointerIds();
    if (ids.length < 2) return 0;
    var a = pointers[ids[0]], b = pointers[ids[1]];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }
  function twoPointerMid() {
    var ids = twoPointerIds();
    var a = pointers[ids[0]], b = pointers[ids[1]];
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  function onKey(e) {
    if (!isOpen) return;
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === '+' || e.key === '=') { animate(); zoomAt(null, null, 1.25); }
    else if (e.key === '-' || e.key === '_') { animate(); zoomAt(null, null, 1 / 1.25); }
    else if (e.key === '0') { animate(); fit(); }
    else if (e.key === '1') { animate(); setScale(1); }
  }

  /* ---------- 变换 ---------- */
  function animate() {
    content.classList.add('pcy-anim');
    clearTimeout(animTimer);
    animTimer = setTimeout(function () { content.classList.remove('pcy-anim'); }, 220);
  }

  function apply() {
    content.style.transform = 'translate3d(' + tx + 'px,' + ty + 'px,0) scale(' + scale + ')';
    scaleEl.textContent = Math.round(scale * 100) + '%';
  }

  function setScale(s) {
    scale = clamp(s, MIN_SCALE, MAX_SCALE);
    apply();
  }

  // 以屏幕坐标 (cx,cy) 为锚点缩放；cx/cy 为空时以舞台中心为锚点
  function zoomAt(cx, cy, factor) {
    var r = stage.getBoundingClientRect();
    var px = (cx == null ? r.left + r.width / 2 : cx) - (r.left + r.width / 2);
    var py = (cy == null ? r.top + r.height / 2 : cy) - (r.top + r.height / 2);
    var ns = clamp(scale * factor, MIN_SCALE, MAX_SCALE);
    if (ns === scale) return;
    var k = ns / scale;
    tx = px - (px - tx) * k;
    ty = py - (py - ty) * k;
    if (Math.abs(ns - fitScale) > 0.001) userZoomed = true;
    scale = ns;
    apply();
  }

  function measureFit() {
    var r = stage.getBoundingClientRect();
    var w = r.width - PADDING, h = r.height - PADDING;
    if (w <= 0 || h <= 0 || baseW <= 0 || baseH <= 0) { fitScale = 1; return; }
    fitScale = clamp(Math.min(w / baseW, h / baseH), MIN_SCALE, FIT_MAX);
  }

  function fit() {
    measureFit();
    scale = fitScale;
    tx = 0; ty = 0;
    userZoomed = false;
    apply();
  }

  /* ---------- 打开 / 关闭 ---------- */
  function measureBase() {
    content.style.transform = 'none';
    var r = content.getBoundingClientRect();
    baseW = r.width || 1;
    baseH = r.height || 1;
  }

  function openContent(node, kind) {
    build();
    lastFocus = document.activeElement;

    content.className = 'pcy-lb-content' + (kind === 'svg' ? ' pcy-for-svg' : ' pcy-for-img');
    content.innerHTML = '';

    var clone = node.cloneNode(true);
    if (kind === 'svg') {
      var r = node.getBoundingClientRect();
      var w = Math.round(r.width) || parseFloat(node.getAttribute('width')) || 800;
      var h = Math.round(r.height) || parseFloat(node.getAttribute('height')) || 600;
      clone.setAttribute('width', w);
      clone.setAttribute('height', h);
      clone.style.width = w + 'px';
      clone.style.height = h + 'px';
      clone.style.maxWidth = 'none';
      // mermaid 的样式作用域是 SVG 自身 id（#mermaid-xxx ...）写在内嵌 <style> 里，
      // 克隆后必须同步改写 id 与 <style> 作用域，否则样式失配、节点退化成黑色填充
      var srcId = node.getAttribute('id');
      var newId = 'pcy-lb-svg-' + Math.random().toString(36).slice(2, 8);
      clone.setAttribute('id', newId);
      if (srcId) {
        var st = clone.querySelector('style');
        if (st && st.textContent.indexOf('#' + srcId) !== -1) {
          st.textContent = st.textContent.split('#' + srcId).join('#' + newId);
        }
      }
    } else {
      clone.removeAttribute('id');
      var nw = node.naturalWidth || node.clientWidth || 800;
      var nh = node.naturalHeight || node.clientHeight || 600;
      clone.style.width = nw + 'px';
      clone.style.height = 'auto';
      clone.style.maxWidth = 'none';
      clone.style.borderRadius = '6px';
      if (nh) clone.setAttribute('height', nh);
    }
    content.appendChild(clone);

    isOpen = true;
    overlay.classList.add('open');
    overlay.setAttribute('aria-hidden', 'false');
    lockScroll(true);

    measureBase();
    fit();
    var btn = $('.pcy-lb-btn[data-act="close"]', bar);
    if (btn) btn.focus();
  }

  function lockScroll(on) {
    if (on) {
      var sw = window.innerWidth - document.documentElement.clientWidth;
      prevBodyPad = document.body.style.paddingRight;
      document.body.style.overflow = 'hidden';
      if (sw > 0) document.body.style.paddingRight = sw + 'px';
    } else {
      document.body.style.overflow = '';
      document.body.style.paddingRight = prevBodyPad;
    }
  }

  function close() {
    if (!isOpen) return;
    isOpen = false;
    overlay.classList.remove('open');
    overlay.setAttribute('aria-hidden', 'true');
    lockScroll(false);
    content.innerHTML = '';
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  /* ---------- 挂载入口 ---------- */
  function attachMermaid(root) {
    var nodes = (root || document).querySelectorAll('.post-body .mermaid');
    Array.prototype.forEach.call(nodes, function (mmd) {
      if (mmd.getAttribute('data-pcy-zoom')) return;
      mmd.setAttribute('data-pcy-zoom', '1');

      // 外层相对定位容器，放全屏按钮（mermaid 自身横向滚动，按钮不能放里面）
      var fig = document.createElement('div');
      fig.className = 'pcy-fig';
      mmd.parentNode.insertBefore(fig, mmd);
      fig.appendChild(mmd);

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pcy-zoom-btn';
      btn.innerHTML = '<span class="pcy-zoom-icon">⛶</span><span>全屏</span>';
      btn.title = '全屏查看（可缩放 / 平移）';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var svg = mmd.querySelector('svg');
        if (svg) openContent(svg, 'svg');
      });
      fig.appendChild(btn);

      mmd.classList.add('pcy-zoomable');
      mmd.title = '点击全屏查看';
      mmd.addEventListener('click', function () {
        var svg = mmd.querySelector('svg');
        if (svg) openContent(svg, 'svg');
      });
    });
  }

  function attachImages(root) {
    var imgs = (root || document).querySelectorAll('.post-body img');
    Array.prototype.forEach.call(imgs, function (img) {
      if (img.getAttribute('data-pcy-zoom')) return;
      img.setAttribute('data-pcy-zoom', '1');
      if (img.classList.contains('no-zoom')) return;

      var wrap = document.createElement('span');
      wrap.className = 'pcy-img-wrap';
      img.parentNode.insertBefore(wrap, img);
      wrap.appendChild(img);

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pcy-zoom-btn';
      btn.innerHTML = '<span class="pcy-zoom-icon">⛶</span><span>全屏</span>';
      btn.title = '全屏查看（可缩放 / 平移）';
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        e.preventDefault();
        openContent(img, 'img');
      });
      wrap.appendChild(btn);

      img.classList.add('pcy-zoomable');
      img.addEventListener('click', function (e) {
        e.preventDefault();
        openContent(img, 'img');
      });

      // 太小的图（图标 / 徽章）不提供缩放
      var tooSmall = function () {
        if (img.naturalWidth && img.naturalWidth < 140) wrap.classList.add('pcy-nozoom');
      };
      if (img.complete) tooSmall();
      else img.addEventListener('load', tooSmall, { once: true });
    });
  }

  function attach(root) {
    try { attachMermaid(root); attachImages(root); } catch (e) { /* 静默失败，不影响阅读 */ }
  }

  window.PCYZoom = { attach: attach, open: openContent, close: close };

  function boot() {
    attach(document);
    // 兜底：图片异步加载 / 内容后续插入时再补一次（轻量，只处理未挂载的）
    setTimeout(function () { attach(document); }, 1200);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
