// Sidebar toggle on smaller screens
document.addEventListener('DOMContentLoaded', function () {
  if (window.innerWidth < 900) {
    document.getElementById('sidebar')?.classList.add('collapsed');
  }
});

// Auto-dismiss flash messages after 5s
setTimeout(function () {
  document.querySelectorAll('.flash-container .alert').forEach(function (el) {
    el.classList.remove('show');
  });
}, 5000);

// Make Plotly charts responsive on resize
window.addEventListener('resize', function () {
  document.querySelectorAll('.chart').forEach(function (el) {
    if (el.data && window.Plotly) {
      Plotly.Plots.resize(el);
    }
  });
}); 