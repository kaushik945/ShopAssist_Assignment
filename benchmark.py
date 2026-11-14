"""Run from repository root: python benchmark.py > reports/optimized.json."""
import json
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline


def main():
    rows = []
    for cls in (SupportCopilot, SupportPipeline):
        for tid in ["T-1001", "T-1002", "T-1003", "T-1004", "T-1091"]:
            agent = cls()  # Cold session: no cache assumptions.
            result = agent.handle_ticket(tid)
            rows.append({
                "track": cls.__name__, "ticket": tid,
                "action": result.action, "kb_article": result.kb_article,
                "points": agent.session.total_points(),
                "calls": [{"label": c.label, "input": c.input_tokens,
                           "cached_input": c.cached_prefix_tokens,
                           "output": c.output_tokens, "points": c.points()}
                          for c in agent.session.calls],
            })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
