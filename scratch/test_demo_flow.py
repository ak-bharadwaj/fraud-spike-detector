from src.web.server import step_demo, reset_demo

reset_demo("M1")
for i in range(12):
    res = step_demo()
    print(f"Window {i:02d} | State: {res['state_machine_status']:<9} | Txs: {res['transaction_count']:<3} | Alerts: {len(res['alerts_emitted'])}")
