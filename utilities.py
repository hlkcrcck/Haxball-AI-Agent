import time, json
import threading

def call_repeatedly(interval, func, *args):
    stopped = threading.Event()
    def loop():
        while not stopped.wait(interval): # the first call is in `interval` secs
            func(*args)
    threading.Thread(target=loop, daemon=True).start()    
    return stopped.set
def wait_for(condition_function, *args):
    start_time = time.time() 
    while time.time() < start_time + 10: 
        if condition_function(*args): 
            return True
        else:
            time.sleep(0.1)
    raise Exception('Timeout waiting for wf')
def press(browser,charkey,duration):
    t1 = threading.Thread(target=hold_key,args=(browser,duration,charkey))
    t1.daemon = True
    t1.start()
    return t1
    #hold_key(browser, duration,charkey) #0.005 , 0.025

def dispatchKeyEvent(driver, name, options = {}):
  options["type"] = name
  body = json.dumps({'cmd': 'Input.dispatchKeyEvent', 'params': options})
  resource = "/session/%s/chromium/send_command" % driver.session_id
  url = driver.command_executor._url + resource
  driver.command_executor._request('POST', url, body)

def hold_key(driver, duration, letter):
  endtime = time.time() + duration
  options = { \
    "code": "Key"+letter,
    "key": letter,
    "text": letter,
    "unmodifiedText": letter,
    "nativeVirtualKeyCode": ord(letter),
    "windowsVirtualKeyCode": ord(letter)
  }

  while True:
    dispatchKeyEvent(driver, "rawKeyDown", options)
    dispatchKeyEvent(driver, "char", options)

    t = threading.currentThread()
    if time.time() > endtime or not getattr(t, "do_run", True):
      dispatchKeyEvent(driver, "keyUp", options)
      break

    options["autoRepeat"] = True
    time.sleep(0.001)

def key_down(driver, letter):
  #time.sleep(0.01)
  options = { \
    "code": "Key"+letter,
    "key": letter,
    "text": letter,
    "unmodifiedText": letter,
    "nativeVirtualKeyCode": ord(letter),
    "windowsVirtualKeyCode": ord(letter)
  }

  dispatchKeyEvent(driver, "rawKeyDown", options)
  dispatchKeyEvent(driver, "char", options)
def key_up(driver, letter):
  #time.sleep(0.0001)
  options = { \
    "code": "Key"+letter,
    "key": letter,
    "text": letter,
    "unmodifiedText": letter,
    "nativeVirtualKeyCode": ord(letter),
    "windowsVirtualKeyCode": ord(letter)
  }

  dispatchKeyEvent(driver, "keyUp", options)
  #options["autoRepeat"] = True
