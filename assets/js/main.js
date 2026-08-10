/* 24th 风格：导航滚动状态 */
(function () {
  var header = document.querySelector('.header');
  if (!header) return;
  var onScroll = function () {
    if (window.scrollY > 8) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
