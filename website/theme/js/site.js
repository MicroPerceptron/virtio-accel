// Two adjustments to mdBook's chrome, both DOM-only:
//
//   1. A brand header at the top of the sidebar linking back to the landing
//      page.
//   2. Relocating the current page's heading tree ("On this page") out of the
//      sidebar chapter list and into a rail beside the content.
//
// (2) is what keeps the sidebar still. mdBook injects the active page's
// headings into the chapter list itself, so the list's height swings with
// whatever page you are on -- 740px here, 2358px on the specification. Every
// navigation therefore lands on a list where the previous scroll position no
// longer exists, and it clamps to the top. Moving the headings out makes the
// chapter list identical on every page, so it simply does not move.
(function () {
  "use strict";

  var WIDE = "(min-width: 1280px)";

  function addBrand() {
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
  }

  // Move the heading tree into a rail beside the content. The nodes are moved,
  // never cloned, so mdBook's scroll-spy -- which looks up `.header-in-summary`
  // globally and toggles a `current-header` class -- keeps working, as do the
  // fold toggles it wired up.
  function relocate(tree) {
    var content = document.getElementById("mdbook-content");
    var main = content && content.querySelector("main");
    if (!content || !main || document.getElementById("va-toc")) {
      return;
    }

    var details = document.createElement("details");
    details.id = "va-toc";

    var summary = document.createElement("summary");
    summary.textContent = "On this page";
    details.appendChild(summary);
    details.appendChild(tree);

    // Before <main> in source order so it reads first without JS and lands in
    // the right grid column with it.
    content.insertBefore(details, main);

    // Open and non-collapsible while the rail has its own column; collapsible
    // and closed once it folds above the content.
    var wide = window.matchMedia(WIDE);
    var sync = function () {
      details.open = wide.matches;
    };
    sync();
    if (wide.addEventListener) {
      wide.addEventListener("change", sync);
    } else if (wide.addListener) {
      wide.addListener(sync);
    }
  }

  // Whether mdBook will produce a heading tree for this page, using its own
  // test (toc.js: h2-h6 inside <main>, carrying an id and an anchor child).
  //
  // This script is the last thing in <body>, so the content is already parsed
  // and this answer is available before the first paint -- which is the point.
  // Reserving the rail's column now, rather than when the tree actually
  // arrives at DOMContentLoaded, keeps the content from visibly narrowing on
  // load, and leaves heading-less pages (the Hexagon operator matrix, whose
  // table wants every pixel) at full width.
  function pageHasHeadings() {
    var main = document.querySelector("#mdbook-content main");
    if (!main) {
      return false;
    }
    var headings = main.querySelectorAll("h2, h3, h4, h5, h6");
    for (var i = 0; i < headings.length; i++) {
      var h = headings[i];
      if (h.id !== "" && h.children.length && h.children[0].tagName === "A") {
        return true;
      }
    }
    return false;
  }

  function reserveColumn() {
    var content = document.getElementById("mdbook-content");
    if (content && pageHasHeadings()) {
      content.classList.add("va-has-toc");
    }
  }

  // mdBook builds the tree in its own DOMContentLoaded handler, so it may not
  // exist yet whichever order the scripts run in. Watch for it, and stop
  // watching once it lands.
  function watchForTree() {
    var sidebar = document.getElementById("mdbook-sidebar");
    if (!sidebar) {
      return;
    }

    var existing = sidebar.querySelector(".on-this-page");
    if (existing) {
      relocate(existing);
      return;
    }

    var observer = new MutationObserver(function () {
      var tree = sidebar.querySelector(".on-this-page");
      if (tree) {
        observer.disconnect();
        relocate(tree);
      }
    });
    observer.observe(sidebar, { childList: true, subtree: true });

    // A page with no headings never gets a tree; do not observe forever.
    window.setTimeout(function () {
      observer.disconnect();
    }, 5000);
  }

  // Hold the chapter list exactly still across a navigation.
  //
  // mdBook has its own version of this, but it adjusts scrollTop by a delta
  // measured before its heading tree is injected, which lands a few tens of
  // pixels out -- enough to clip the chapter you just clicked. With the
  // headings moved to the rail the list is identical on every page, so the
  // position it should end up at is exactly computable. This runs from the end
  // of <body>, after mdBook's attempt, and is therefore the one that sticks.
  var SCROLL_KEY = "va-sidebar-offset";

  function keepSidebarStill() {
    var box = document.querySelector("#mdbook-sidebar .sidebar-scrollbox");
    if (!box) {
      return;
    }

    box.addEventListener(
      "click",
      function (event) {
        var link = event.target.closest && event.target.closest("a");
        if (!link || !box.contains(link)) {
          return;
        }
        var href = link.getAttribute("href") || "";
        if (!href || href.charAt(0) === "#") {
          return;
        }
        var offset = link.getBoundingClientRect().top - box.getBoundingClientRect().top;
        try {
          sessionStorage.setItem(SCROLL_KEY, String(offset));
        } catch (e) {
          /* private mode: fall back to mdBook's behaviour */
        }
      },
      { passive: true }
    );

    var saved;
    try {
      saved = sessionStorage.getItem(SCROLL_KEY);
      sessionStorage.removeItem(SCROLL_KEY);
    } catch (e) {
      return;
    }

    var active = box.querySelector(".active");
    if (!active) {
      return;
    }

    if (saved !== null) {
      // Put the clicked chapter back where it was on the previous page.
      var top = active.getBoundingClientRect().top - box.getBoundingClientRect().top;
      box.scrollTop = box.scrollTop + top - parseFloat(saved);
    }

    // Whatever the maths or the clamp produced, do not leave it half cut off.
    var a = active.getBoundingClientRect();
    var b = box.getBoundingClientRect();
    if (a.bottom > b.bottom) {
      box.scrollTop += a.bottom - b.bottom + 8;
    } else if (a.top < b.top) {
      box.scrollTop -= b.top - a.top + 8;
    }
  }

  addBrand();
  reserveColumn();
  watchForTree();
  keepSidebarStill();
})();
