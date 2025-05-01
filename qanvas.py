import curses
import datetime
import time

from MongoDB_code.main import main as update_database, get_last_updated_time
# from Agent_code.dummy import main as start_chat #TODO
from Agent_code.main import main as start_chat # Done

last_updated_time = get_last_updated_time()

def draw_menu(stdscr, last_updated_time):
    stdscr.clear()
    height, width = stdscr.getmaxyx()
    curses.curs_set(0)
    title_lines = [
        "   ___      _                         ",
        "  / _ \    / \   _ ____   ____ _ ___  ",
        " | | | |  / _ \ | '_ \ \ / / _` / __| ",
        " | |_| | / ___ \| | | \ V / (_| \__ \ ",
        "  \__\_\/_/   \_\_| |_|\_/ \__,_|___/ ",                                
    ]
    start_y = 1
    start_x = 2
    for i, line in enumerate(title_lines):
        stdscr.addstr(start_y + i, start_x, line, curses.A_BOLD)

    # stdscr.addstr(1, (width - len(title)) // 2, title, curses.A_BOLD)

    stdscr.addstr(7, 4, "[R] Update database")
    stdscr.addstr(8, 4, "[C] Start chat")
    stdscr.addstr(9, 4, "[Q] Quit")

    if last_updated_time:
        timestamp_str = last_updated_time.strftime("%Y-%m-%d %H:%M:%S")
        status_line = f"Last updated at: {timestamp_str}"
        stdscr.addstr(height - 1, 4, status_line[:width - 1], curses.A_REVERSE)
    else:
        stdscr.addstr(height - 1, 4, "Last updated at: --", curses.A_REVERSE)

    stdscr.refresh()

def run_database_update(stdscr):
    global last_updated_time
    stdscr.clear()
    stdscr.addstr(1, 2, "[INFO] Updating database...")
    stdscr.refresh()
    try:
        update_database()
        stdscr.addstr(3, 2, "[SUCCESS] Database update complete.")
        last_updated_time = datetime.datetime.now()
    except Exception as e:
        stdscr.addstr(3, 2, f"[ERROR] {str(e)}")
    stdscr.addstr(5, 2, "Press any key to return to menu.")
    stdscr.getch()

def run_chat(stdscr):
    stdscr.clear()
    curses.curs_set(2)
    if not last_updated_time:
        stdscr.addstr(1, 2, "[ERROR] Database not updated. Please update first.")
        stdscr.getch()
        return
    stdscr.addstr(0, 2, "--- Chat Mode (type 'exit' or 'quit' to leave) ---")
    stdscr.refresh()

    chat_win = curses.newwin(curses.LINES - 4, curses.COLS - 2, 1, 1)
    input_win = curses.newwin(3, curses.COLS - 2, curses.LINES - 3, 1)

    chat_history = []
    while True:
        chat_win.clear()
        input_win.clear()

        max_lines = curses.LINES - 6
        for idx, (sender, msg) in enumerate(chat_history[-max_lines:]):
            chat_win.addstr(idx+2, 2, f"{sender}: {msg}")
        chat_win.box()
        chat_win.refresh()

        input_win.addstr(0, 0, "You: ")
        input_win.refresh()
        curses.echo()
        user_input = input_win.getstr(1, 0, 100).decode("utf-8").strip()
        
        curses.noecho()

        if user_input.lower() in ['exit', 'quit']:
            break
        # connect to the agent
        try:
            response = start_chat(user_input)
        except Exception as e:
            response = f"[ERROR] {str(e)}"

        chat_history.append(("You", user_input))
        chat_history.append(("QAnvas", response))

def main(stdscr):
    global last_updated_time
    curses.curs_set(0)
    stdscr.nodelay(False)
    while True:
        draw_menu(stdscr, last_updated_time)
        key = stdscr.getch()

        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('r'), ord('R')]:
            run_database_update(stdscr)
        elif key in [ord('c'), ord('C')]:
            run_chat(stdscr)

if __name__ == "__main__":
    curses.wrapper(main)