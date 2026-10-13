#!/usr/bin/env python3
"""
pong_display.py — watch the dye-reservoir lane pong (run on your laptop, not the Pi).

    pip install paho-mqtt pygame            # pygame optional: use --text without it
    python pong_display.py --host <pi-address> --unit pioreactor01 --experiment "Demo experiment"
    python pong_display.py --sim            # no hardware: a toy reservoir, to see the game

Shows the ball (where the dye was dosed), a hollow marker for the lane it was aimed at, and the paddle
(the readout's guess, scored next slot).  A running "copy the last ball lane" paddle is scored on the
same slots, so you can see live whether the dye's memory is beating a paddle with no memory.
It only listens; it never talks to the pumps.
"""
import argparse
import json
import random
import sys
import threading
import time

try:
    import pygame
except ImportError:
    pygame = None
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


# identical copy of make_ball in reservoir_pong.py / fit_pong_readout.py (only used by --sim)
def make_ball(n, n_lanes=3, seed=0, kind="wander", hold=20.0, jitter=0.6, start_lane=0, start_dir=1):
    rng = random.Random(seed)
    lanes, aims = [], []
    if kind == "zigzag":
        lane, d = int(start_lane), int(start_dir)
        for _ in range(n):
            lanes.append(lane)
            nxt = lane + d
            if nxt < 0 or nxt >= n_lanes:
                d = -d
                nxt = lane + d
            lane = nxt
        return lanes, list(lanes)
    target = rng.randrange(n_lanes)
    for _ in range(n):
        if rng.random() < 1.0 / max(1.0, float(hold)):
            target = rng.randrange(n_lanes)
        lane = target
        if rng.random() < jitter:
            lane = min(n_lanes - 1, max(0, target + rng.choice((-1, 1))))
        lanes.append(lane)
        aims.append(target)
    return lanes, aims


class Feed:
    """Thread-safe list of per-slot payloads from the plugin (or the simulator)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []
        self.label = "WAITING"

    def add(self, row):
        with self.lock:
            if not self.rows or row.get("slot") != self.rows[-1].get("slot"):
                self.rows.append(row)

    def snapshot(self):
        with self.lock:
            return list(self.rows), self.label


def start_live(feed, host, port, unit, experiment, user="pioreactor", password="raspberry"):
    if mqtt is None:
        sys.exit("pip install paho-mqtt")
    topic = f"pioreactor/{unit}/{experiment}/reservoir_pong/last_slot"
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        client = mqtt.Client()

    if user and password:
        client.username_pw_set(user, password)

    def on_connect(c, userdata, flags, rc, *args):
        if rc == 0:
            print(f"Connected to MQTT broker at {host}:{port} (Auth OK)!")
            c.subscribe(topic)
            c.subscribe("pioreactor/+/+/reservoir_pong/last_slot")
            c.subscribe("pioreactor/#")
        else:
            print(f"Connection refused by broker: rc={rc}")

    def on_message(_c, _u, msg):
        try:
            if not (msg.topic.endswith("/last_slot") or "reservoir_pong/last_slot" in msg.topic):
                return
            row = json.loads(msg.payload.decode())
            print(f"--> [LIVE] Slot {row.get('slot')} from '{msg.topic}': ball={row.get('lane')} -> paddle={row.get('pred_next')} | hits={row.get('hits')}/{row.get('scored')}")
            feed.label = row.get("mode", "play").upper()
            feed.add(row)
        except Exception as e:
            print("Error parsing MQTT payload:", e)

    client.on_connect, client.on_message = on_connect, on_message
    client.connect(host, port, 60)
    client.loop_start()
    print(f"Connecting to '{host}:{port}' with user '{user}'...")


def start_sim(feed, interval, seed, n_lanes=3, sigma=0.12, lam=0.825):
    feed.label = "SIMULATED"
    lanes, aims = make_ball(100000, n_lanes, seed)

    def run():
        rng, e, pending, hits, scored = random.Random(seed + 1), 0.0, None, 0, 0
        for n, (lane, aim) in enumerate(zip(lanes, aims)):
            e = lam * e + (1 - lam) * lane
            hit = -1
            if pending is not None:
                hit = int(pending == aim)
                scored += 1
                hits += hit
            pending = int(min(n_lanes - 1, max(0, round(e + rng.gauss(0, sigma)))))
            feed.add({"slot": n, "lane": lane, "aim": aim, "pred_next": pending, "hit_prev": hit,
                      "hits": hits, "scored": scored, "n_lanes": n_lanes, "mode": "sim"})
            time.sleep(interval)

    threading.Thread(target=run, daemon=True).start()


def tallies(rows):
    """Reservoir paddle vs a memoryless 'copy the last ball lane' paddle on the same slots."""
    res_h = res_n = cp_h = cp_n = 0
    for prev, row in zip([None] + rows[:-1], rows):
        if prev is None or row.get("aim") is None:
            continue
        cp_n += 1
        cp_h += int(prev["lane"] == row["aim"])
        if row.get("hit_prev", -1) >= 0:
            res_n += 1
            res_h += row["hit_prev"]
    return res_h, res_n, cp_h, cp_n


def text_loop(feed):
    seen = 0
    while True:
        rows, label = feed.snapshot()
        for r in rows[seen:]:
            h, n, ch, cn = tallies(rows[: rows.index(r) + 1])
            print(f"[{label}] slot {r['slot']:4d} ball={r['lane']} aim={r.get('aim')} paddle->{r['pred_next']} "
                  f"| reservoir {h}/{n} | copy-lane {ch}/{cn}")
        seen = len(rows)
        time.sleep(0.5)


def gui_loop(feed):
    pygame.init()
    W, H, TRAIL = 980, 440, 24
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Reservoir lane pong")
    font, big = pygame.font.SysFont("monospace", 16), pygame.font.SysFont("monospace", 22, bold=True)
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
        rows, label = feed.snapshot()
        screen.fill((14, 16, 22))
        n_lanes = rows[-1]["n_lanes"] if rows else 3
        ys = [110 + i * (H - 220) // max(1, n_lanes - 1) for i in range(n_lanes)]
        for y in ys:
            pygame.draw.line(screen, (45, 50, 62), (40, y), (W - 80, y), 1)
        for k, row in enumerate(rows[-TRAIL:]):
            x = 60 + k * 34
            lane = row["lane"]
            if row.get("aim") is not None:
                pygame.draw.rect(screen, (90, 140, 255), (x - 9, ys[row["aim"]] - 9, 18, 18), 2)
            newest = k == len(rows[-TRAIL:]) - 1
            pygame.draw.circle(screen, (255, 230, 90) if newest else (140, 130, 70), (x, ys[lane]), 9 if newest else 5)
        if rows:
            last = rows[-1]
            pred = last["pred_next"]
            if pred is not None and pred >= 0:
                ok = last.get("hit_prev", -1)
                col = (80, 220, 120) if ok == 1 else (230, 90, 90) if ok == 0 else (200, 200, 200)
                pygame.draw.rect(screen, col, (W - 70, ys[pred] - 30, 18, 60), border_radius=4)
            h, n, ch, cn = tallies(rows)
            screen.blit(big.render(f"[{label}]  slot {last['slot']}", True, (230, 230, 240)), (40, 20))
            screen.blit(font.render(f"reservoir paddle : {h}/{n} = {100 * h / n:.0f}%" if n else "reservoir paddle : -", True, (120, 230, 150)), (40, H - 80))
            screen.blit(font.render(f"copy-last-lane   : {ch}/{cn} = {100 * ch / cn:.0f}%  (no memory)" if cn else "", True, (200, 200, 120)), (40, H - 56))
            screen.blit(font.render("yellow=ball  blue box=aimed lane  bar=paddle (green hit / red miss)", True, (120, 125, 140)), (40, H - 28))
        pygame.display.flip()
        clock.tick(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost", help="MQTT broker (usually the leader Pioreactor)")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--unit", default="pioreactor01")
    ap.add_argument("--experiment", default="Demo experiment")
    ap.add_argument("--sim", action="store_true", help="toy simulated reservoir, no hardware")
    ap.add_argument("--sim-interval", type=float, default=0.6, help="seconds per simulated slot")
    ap.add_argument("--sim-seed", type=int, default=11)
    ap.add_argument("--text", action="store_true", help="plain text instead of a window")
    ap.add_argument("--user", default="pioreactor", help="MQTT username (default: pioreactor)")
    ap.add_argument("--password", default="raspberry", help="MQTT password (default: raspberry)")
    a = ap.parse_args()
    feed = Feed()
    if a.sim:
        start_sim(feed, a.sim_interval, a.sim_seed)
    else:
        start_live(feed, a.host, a.port, a.unit, a.experiment, user=a.user, password=a.password)
    if a.text or pygame is None:
        if pygame is None and not a.text:
            print("pygame not installed -> text mode (pip install pygame for the window)")
        text_loop(feed)
    else:
        gui_loop(feed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
