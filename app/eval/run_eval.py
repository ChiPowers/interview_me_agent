import yaml
from agent.lc_controller import LCController

def run():
    ctl = LCController()
    with open("eval/qas.yaml") as f:
        data = yaml.safe_load(f)
    for item in data.get("questions", []):
        res = ctl.respond(item["q"])
        print("Q:", item["q"])
        print("A:", res["answer"])
        print("Refs:\n", res.get("footnotes", ""))
        print("----\n")

if __name__ == "__main__":
    run()