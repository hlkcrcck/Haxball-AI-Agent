import utilities as ut
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
class Player:
    def __init__(self, name, room):
        self.score = 0
        self.room = room
        self.name = name
        self.thread = None
        self.thread2 = None
        self.id = ""
        self.key_downs = []
        self.pos = {'x':-200.0, 'y':0.0}
        self.team = ""
        self.browser = webdriver.Chrome()
        self.browser.get(room.roomLink)
        ut.wait_for(self.user_name_enter)
        ut.wait_for(self.getId)
    def move_to_team(self, team_ind):
        js = "window.room.setPlayerTeam("+str(self.id)+","+str(team_ind)+");"
        self.room.browser.execute_script(js)
        if team_ind == 1:
            self.team = "red"
        else:
            self.team = "blue"
    def getId(self):
        jsGetId = 'var pl = window.room.getPlayerList();for (var i = 0; i < pl.length; i++) { if (pl[i].name == "'+self.name+'"){return pl[i].id; }}'
        self.id = self.room.browser.execute_script(jsGetId)
        if self.id is None:
            return False
        else:
            return True
    def getPosition(self):
        jsGetPos = "return window.room.getPlayer("+str(self.id)+").position"
        self.pos = self.room.browser.execute_script(jsGetPos)
        return self.pos
    def user_name_enter(self):
        jsUserName = 'document.getElementsByTagName("iframe")[0].contentDocument.getElementsByTagName("input")[0].value = "'+self.name+'";'
        jsButtonEnable = 'document.getElementsByTagName("iframe")[0].contentDocument.getElementsByTagName("button")[0].disabled = false;'
        jsClick = 'document.getElementsByTagName("iframe")[0].contentDocument.getElementsByTagName("button")[0].click();'
        try:
            self.browser.execute_script(jsUserName)
            self.browser.execute_script(jsButtonEnable)
            self.browser.execute_script(jsClick)
            return True
        except:
            return False
    def action(self, act):
        ut.hold_key(self.browser, self.room.possible_actions[act][1], self.room.possible_actions[act][0])
    def action_physical(self, act):
        ut.hold_key(self.browser, 0.1, self.room.possible_actions[act][0])
    def threaded_action(self, act):
        if(self.thread is not None):
            self.thread.do_run = False
        self.thread = ut.press(self.browser, self.room.possible_actions[act][1], self.room.possible_actions[act][0])
    def start_physical_action(self):
        js = "return window.room.getBallPosition()"
        prev_ball_pos = self.room.ball_pos
        self.room.ball_pos = self.room.browser.execute_script(js)
        self.getPosition()
        velocity_x = self.room.ball_pos['x'] - prev_ball_pos['x']
        velocity_y = self.room.ball_pos['y'] - prev_ball_pos['y']
        if self.team == "red":
            dif_x = ((self.room.ball_pos['x']-10) - self.pos['x'])
            dif_y = (self.room.ball_pos['y'] - self.pos['y'])
            if abs(dif_x) < 16 and abs(dif_y) < 16:
                self.action_physical(0)
            if dif_x > 0:
                self.action_physical(2)
            else:
                self.action_physical(3)
            if dif_y > 0:
                self.action_physical(1)
            else:
                self.action_physical(4)
        else:
            dif_x = ((self.room.ball_pos['x']+10) - self.pos['x'])
            dif_y = (self.room.ball_pos['y'] - self.pos['y'])
            if abs(dif_x) < 16 and abs(dif_y) < 16:
                self.action_physical(0)
            if dif_x > 0:
                self.action_physical(3)
            else:
                self.action_physical(2)
            if dif_y > 0:
                self.action_physical(4)
            else:
                self.action_physical(1)
    def act_key_down(self, act):
        ut.key_down(self.browser, self.room.possible_actions[act][0])
        self.key_downs.append(act)
    def start_physical(self):
        while 1==1:
            js = "return window.room.getBallPosition()"
            prev_ball_pos = self.room.ball_pos
            self.room.ball_pos = self.room.browser.execute_script(js)
            self.getPosition()
            ball_velocity_x = (self.room.ball_pos['x'] - prev_ball_pos['x'])*5
            ball_velocity_y = self.room.ball_pos['y'] - prev_ball_pos['y']
            predicted_ball_pos = {'x':ball_velocity_x+self.room.ball_pos['x'],'y':ball_velocity_y+self.room.ball_pos['y']}
            for i in self.key_downs:
                ut.key_up(self.browser, self.room.possible_actions[i][0])
            self.key_downs = []
            if self.team == "red":
                dif_x = ((predicted_ball_pos['x']-10) - self.pos['x'])
                dif_y = (predicted_ball_pos['y'] - self.pos['y'])
                if abs(dif_x) < 16 and abs(dif_y) < 16:
                    self.act_key_down(0)
                if dif_x > 0:
                    self.act_key_down(2)
                else:
                    self.act_key_down(3)
                if dif_y > 20:
                    self.act_key_down(1)
                elif dif_y < -20:
                    self.act_key_down(4)
            else:
                dif_x = ((predicted_ball_pos['x']+10) - self.pos['x'])
                dif_y = (predicted_ball_pos['y'] - self.pos['y'])
                if abs(dif_x) < 16 and abs(dif_y) < 16:
                    self.act_key_down(0)
                if dif_x > 0:
                    self.act_key_down(2)
                else:
                    self.act_key_down(3)
                if dif_y > 20:
                    self.act_key_down(1)
                elif dif_y < -20:
                    self.act_key_down(4)
    def update_score(self):
        js = "return window.room.getScores()"
        oldscore = self.score
        score_obj = self.room.browser.execute_script(js)
        self.score = score_obj[self.team]
        self.room.time = score_obj['time']
        return self.score - oldscore