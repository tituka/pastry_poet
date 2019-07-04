import schedule
import time
import puu

schedule.every(10).minutes.do(puu.train_cycle)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute