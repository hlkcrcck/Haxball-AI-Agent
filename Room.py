import utilities as ut
from Player import Player
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import threading
class Room:
    def __init__(self,name,token,number_of_bots):
        self.press_time = {'short_press':0.005, 'long_press':0.1}
        self.possible_keys = ['X','S','D','A','W']
        self.possible_actions = [[ch, self.press_time['long_press']] for ch in self.possible_keys]
        self.roomLink = ''
        self.browser = webdriver.Chrome()
        self.browser.get('https://html5.haxball.com/headless')
        ut.wait_for(self.start_room, token)
        ut.wait_for(self.get_room_link)
        self.bots = []
        self.curr_screen = -1
        self.time = 0
        self.cancel_future_calls = None
        self.steps = 0
        self.maxsteps = 64
        self.ball_pos = {'x':0.0, 'y':0.0}
        for i in range(number_of_bots):
            pname = "P"+str(i)
            self.bots.append(Player(pname,self))
    def reward_for(self,bot_ind,opp_ind):
        reward = 0.2
        """js = "return window.lastplayerkicked"
        last_player_kicked = self.browser.execute_script(js)
        if last_player_kicked == self.bots[bot_ind].id:
            reward += 1
        elif last_player_kicked == self.bots[opp_ind].id:
            reward -= 0.5
        js = "window.lastplayerkicked = 0"
        self.browser.execute_script(js)
        if self.time > 0.0:
            reward += 0.1"""
        #reward -= abs(self.bots[bot_ind].pos['x'] - self.ball_pos['x'])/1000
        #reward -= abs(self.bots[bot_ind].pos['y'] - self.ball_pos['y'])/1000
        reward += self.bots[bot_ind].update_score()*5
        reward -= self.bots[opp_ind].update_score()*5
        """if(self.bots[bot_ind].score < self.bots[opp_ind].score):
            reward -= 0.3
        elif (self.bots[bot_ind].score == self.bots[opp_ind].score):
            reward -= 0.01
        else:
            reward += 0.3"""
        return reward
    def step(self, action):
        self.steps += 1
        self.bots[1].action(action)
        js = "return window.gamefinished"
        is_game_finished = self.browser.execute_script(js)
        if (self.bots[0].score > self.bots[1].score or is_game_finished is 1 or self.bots[1].score > self.bots[0].score):
            js = "window.gamefinished = 0"
            self.browser.execute_script(js)
            return self.reward_for(1, 0), True
        """if self.steps == self.maxsteps:
            self.steps = 0
            return self.reward_for(1, 0), True"""
        return self.reward_for(1, 0), False
    def get_screen(self):
        js = "return window.room.getBallPosition()"
        self.ball_pos = self.browser.execute_script(js)
        pos1 = self.bots[0].getPosition()
        pos2 = self.bots[1].getPosition()
        """current = [pos1['x']/300, pos1['y']/150, pos2['x']/300, pos2['y']/150, self.ball_pos['x']/300, self.ball_pos['y']/150]
        if self.curr_screen == -1:
            self.curr_screen = current*4
        else:
            [self.curr_screen.pop(0) for i in range(6)]
            [self.curr_screen.append(i) for i in current]"""
        self.curr_screen = [(pos1['x']+400)/800, (pos1['y']+200)/400, (pos2['x']+400)/800, (pos2['y']+200)/400, (self.ball_pos['x']+350)/700, (self.ball_pos['y']+200)/400]
        return self.curr_screen
    def reset(self):
        self.stopGame()
        self.startGame()
    def startGame(self):
        self.bots[0].move_to_team(1)
        self.bots[1].move_to_team(2)
        jsstartGame = 'window.room.startGame();'
        self.browser.execute_script(jsstartGame)
        #self.cancel_future_calls = ut.call_repeatedly(0.01, self.bots[0].start_physical_action)
        t1 = threading.Thread(target=self.bots[0].start_physical)
        t1.daemon = True
        t1.start()
        t2 = threading.Thread(target=self.bots[1].start_physical)
        t2.daemon = True
        t2.start()
    def stopGame(self):
        """if self.cancel_future_calls is not None:
            self.cancel_future_calls()"""
        jsstopGame = 'window.room.stopGame();'
        self.browser.execute_script(jsstopGame)
    def start_room(self,token):
        jsStartRoom = 'window.room = HBInit({token:"'+token+"""",roomName: "AI", maxPlayers: 8});
        window.room.setTeamsLock(true);
        window.gamefinished = 0;
        window.lastplayerkicked = 0;
        function handleVictory(scores) {
	        window.gamefinished = 1;
        }
        function handleKick(player) {
	        window.lastplayerkicked = player.id;
        }
        room.onTeamVictory = handleVictory;
        room.onPlayerBallKick = handleKick;
        """
        try:
            self.browser.execute_script(jsStartRoom)
            return True
        except:
            return False
    def get_room_link(self):
        jsGetRoomLink = 'return document.getElementsByTagName("iframe")[0].contentDocument.getElementById("roomlink").getElementsByTagName("a")[0].getAttribute("href")'
        try:
            self.roomLink = self.browser.execute_script(jsGetRoomLink)
            return True
        except:
            return False

"""from PIL import Image

driver = webdriver.Chrome()
driver.get('https://www.google.co.in')

element = driver.find_element_by_id("lst-ib")

location = element.location
size = element.size

driver.save_screenshot("shot.png")

x = location['x']
y = location['y']
w = size['width']
h = size['height']
width = x + w
height = y + h

im = Image.open('shot.png')
im = im.crop((int(x), int(y), int(width), int(height)))
im.save('image.png')"""