PREDICTIVE RELAY DEPLOYMENT DASHBOARD

Files:
  dashboard/index.html - dashboard page
  dashboard/style.css - styling
  dashboard/app.js - CSV import, metric detection and charts
  dashboard/data/ - reserved for generated JSON exports

INSTALL INTO YOUR PROJECT:
1. Copy the dashboard folder into:
   C:\Users\Admin\Downloads\proj\wireless_relay comm\predictive-relay-deployment\dashboard
2. Open PowerShell in the project root:
   cd "C:\Users\Admin\Downloads\proj\wireless_relay comm\predictive-relay-deployment"
3. Start a local static server (recommended):
   python -m http.server 8000
4. Open in browser:
   http://localhost:8000/dashboard/
5. In the dashboard, click Import simulation CSVs and select actual summary CSV files from data\processed\script32, script35 and script36.

Notes:
- The dashboard does not fabricate results and does not rerun simulations.
- It reads selected CSV files in the browser. No file contents are sent to a server.
- Chart auto-detection is heuristic. Verify the detected metric/policy columns against your actual CSV headers before presenting results.
- Chart.js is loaded from a CDN, so an internet connection is needed for charts. The rest of the page loads locally.
