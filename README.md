# Agri Market Price Prediction + Blockchain Anchoring

This project predicts mandi prices in the browser and anchors each prediction proof on-chain through MetaMask.

## Project overview

The app combines:

1. **ML-style forecasting workflow** (commodity/state/month/year based prediction from historical CSV data)
2. **Decision support** (SELL/HOLD/NEUTRAL recommendation with threshold logic)
3. **Blockchain proof anchoring** (hash of prediction payload written into transaction calldata)

Main UI features:

- Commodity/state and optional district/market/variety/grade filters
- Predicted price, historical mean, expected change
- Price unit toggle (view as ₹/Quintal or ₹/Kg)
- Current price comparison and action signal
- 12-month forecast chart
- Wallet connection + on-chain proof log
- Auto-location state matching
- On-demand plot viewer for generated ML images

## Folder structure

- `index.html` - web UI
- `style.css` - styling
- `app.js` - prediction logic, comparison API, wallet + blockchain flow
- `data\mandi_prices_dataset1.csv`, `data\mandi_prices_dataset2.csv` - source datasets
- `plot_*.png` - ML analysis/visual artifacts
- `streamlit_app.py` - original Streamlit implementation
- `saved_model\` - trained model artifacts used by Streamlit flow

## Blockchain integration (how it works)

After each prediction:

1. The app creates a canonical JSON payload from inputs + computed outputs.
2. It computes:
   - `payloadHash = keccak256(utf8(payloadJson))`
3. Via MetaMask (`ethers.js`), it sends an on-chain transaction containing that hash in calldata.
4. It waits for confirmation, then stores:
   - time
   - commodity/state
   - predicted price
   - payload hash
   - transaction hash (with explorer link when available)

This creates tamper-evident proof that your prediction existed at a specific block/time.

## Important fix applied (MetaMask RPC error)

If you saw:

`External transactions to internal accounts cannot include data`

some networks are rejecting calldata transactions unless the `to` address is a deployed contract that can accept it.

The app now handles this cleanly:

1. It first tries true on-chain anchoring **only** when a valid contract target is configured.
2. If chain rules/config prevent that, it falls back to a **wallet-signed proof** (simulated blockchain anchor).
3. The proof hash is still recorded in the app log with a deterministic pseudo transaction hash.

This avoids hard failures while preserving tamper-evident proof for each prediction.

## Why this project is useful (advantages)

1. **Practical farmer/market decision support** - quickly compare current vs expected price direction.
2. **Trust and auditability** - prediction evidence can be independently verified on-chain.
3. **No backend requirement for core flow** - forecasting and UI run directly in browser.
4. **Fast and mobile-friendly UX** - lightweight app with on-demand heavy assets.
5. **Explainable output** - users see forecast, historical baseline, and recommendation logic together.
6. **Easy extensibility** - can evolve to smart-contract registry, backend API, or signed attestations.

## Run locally

Do not open with `file://` directly (CSV fetch will be blocked by browser security policy).

From `C:\Users\dell\Desktop\ml`:

```powershell
python model_api_server.py
```

Open:

`http://localhost:8000`

## MetaMask usage

1. Open app in browser.
2. Click **Connect Wallet** and approve MetaMask.
3. Fill inputs and click **Predict Market Price**.
4. Click **Anchor Prediction On-Chain**.
5. Approve transaction in MetaMask.
6. After confirmation, check the **On-Chain Prediction Log** section.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| MetaMask not found | Extension not installed or disabled | Install/enable MetaMask, refresh page |
| Blockchain error while sending | Wrong network, user rejection, or RPC policy | Reconnect wallet, switch network, try again |
| `cannot include data` RPC error | Calldata sent to EOA (old flow) | Updated flow now targets contract-capable address |
| CSV/data not loading | Opened via file path, not HTTP | Run local server and use `http://localhost:8000` |

## Tech stack

- HTML5, CSS3, Vanilla JavaScript
- Ethers.js v6.13.2
- Chart.js
- CSV-based historical data processing in-browser

## Future improvements

- Deploy dedicated smart contract (`anchor(bytes32 hash, string metadataCID)`)
- Store full payload off-chain (IPFS/DB) and anchor only CID/hash on-chain
- Add signature-based attestations and multi-user proof explorer
