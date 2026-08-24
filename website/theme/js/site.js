// Adds a brand header to the top of the mdBook sidebar linking back to the
// landing page. The <nav id="mdbook-sidebar"> element is in the static HTML
// (only its table of contents is populated later by book.js), so this runs
// without waiting on anything.
(function () {
  "use strict";

  var sidebar = document.getElementById("mdbook-sidebar");
  if (!sidebar || sidebar.querySelector(".va-brand")) {
    return;
  }

  var brand = document.createElement("a");
  brand.className = "va-brand";
  brand.href = "/";
  brand.setAttribute("aria-label", "virtio-accel home");
  brand.innerHTML =
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 3 3 7.5v9L12 21l9-4.5v-9L12 3Z"/>' +
    '<path d="M12 12 3 7.5M12 12l9-4.5M12 12v9" opacity="0.55"/>' +
    "</svg><b>virtio-accel</b>";

  sidebar.insertBefore(brand, sidebar.firstChild);
})();
