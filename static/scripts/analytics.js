"use strict";

const detail = document.querySelector("[data-analytics-click]");
const results = document.querySelector("[data-analytics-results]");

window.addEventListener("pageshow", () => {
    if (detail) {
        try {
            sessionStorage.setItem(
                "irwa:dwell:" + detail.dataset.searchId, detail.dataset.analyticsClick
            );
        } catch {
            // The Back to results link records a return even if storage is disabled.
        }
    }

    if (results) {
        const searchId = results.dataset.searchId;
        const url = new URL(window.location.href);
        if (url.searchParams.get("search_id") !== searchId || url.searchParams.has("returned_click")) {
            url.searchParams.set("search_id", searchId);
            url.searchParams.delete("returned_click");
            window.history.replaceState(window.history.state, "", url);
        }

        try {
            const key = "irwa:dwell:" + searchId;
            const clickId = sessionStorage.getItem(key);
            if (clickId) {
                fetch(results.dataset.returnUrl, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({click_id: clickId, search_id: searchId}),
                    keepalive: true,
                }).then((response) => {
                    // A 404 means the click expired or no longer exists.
                    if ((response.ok || response.status === 404) && sessionStorage.getItem(key) === clickId) {
                        sessionStorage.removeItem(key);
                    }
                }).catch(() => {
                    // Keep the pending return for the next visit to these results.
                });
            }
        } catch {
            // Search results remain available when storage is disabled.
        }
    }
});
