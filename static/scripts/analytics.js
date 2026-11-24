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
                sessionStorage.removeItem(key);
                fetch(results.dataset.returnUrl, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({click_id: clickId, search_id: searchId}),
                    keepalive: true,
                }).catch(() => {
                    // Dwell tracking is optional for the shopper's navigation.
                });
            }
        } catch {
            // Search results remain available when storage is disabled.
        }
    }
});
