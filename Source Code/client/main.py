'''
Fact U App - A mobile app for news verification using hidden markov model and sentence similary analysis

Emmanuel B. Constantino Jr.
2013-08147

Homer C. Malijan
2013-09022
'''
#imports for rendering front end
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.factory import Factory
from kivy.animation import Animation

import time
import threading
import socket

# load kv language for GUI
Builder.load_file('main.kv')

# main app
class FactU(FloatLayout):
    inputString = ''
    inputIP = ''
    host = ''
    trial = 1
    counter = 1
    anim_bar = Factory.AnimWidget()
    anim = Animation(opacity=0.3, width=100, duration=0.6)

    def update(self):
        threading.Thread(target=self.start_test).start()
        self.ids.submit_button.disable = True

    #get ip from Text Input
    def getIP(self):
        self.host = self.inputIP
        self.remove_widget(self.ids.ipPopup)

    def start_test(self):
        #default port for this application
        port = 8081

        #connect to sever via input IP address and port 8081
        s = socket.socket()
        s.connect((self.host, port))
        #send input string to server
        s.send(str.encode(str(self.inputString)))
        data = str.encode("Start")

        query = str(self.inputString)

        #create loading animation at the bottom of the UI
        if self.counter == 1:
            self.anim_box.add_widget(self.anim_bar)
            self.anim += Animation(opacity=1, width=400, duration=0.8)
            self.counter += 1

        self.anim.repeat = True
        self.anim.start(self.anim_bar)

        #receive outpur from server
        data = s.recv(4096)

        #post process output for rendering in the front end
        split1 = "', '"
        if split1 in data.decode():
            tempResult = data.decode().split(split1)
        else:
            tempResult = data.decode().split("', \"")


        l = list(str(tempResult[0]))
        del(l[1])
        del(l[0])
        resA = "".join(l) + "\n\n\n"
        print("resA - ", resA)

        if "Insufficient" not in resA:
            l = list(str(tempResult[1]))
            del(l[len(tempResult[1])-1])
            del(l[len(tempResult[1])-2])
            del(l[len(tempResult[1])-3])
            del(l[len(tempResult[1])-4])
            resB = "".join(l)

            print("resB - ", resB)
            checker = '||'
            resultHolder = resA
            resA = ''
            count = 0
            try:
                 if checker in resB:
                     resultList = resB.split('||')
                     for singleResult in resultList:
                         if count==3:
                             break
                         tempResultHolder = singleResult.split('***')
                         if tempResultHolder[0] not in resA:
                             count+=1
                             resA += "[size=30][b]" + tempResultHolder[0] + "[/b][/size]" + "\n" + "[size=30][i]" + tempResultHolder[1] + "[/i][/size]"+ "\n\n"
                         print(tempResultHolder[0], ' - ', tempResultHolder[1])
                 else:
                     tempResultHolder = resB.split('***')
                     if tempResultHolder[0] not in resA:
                         count+=1
                         resA += "[size=30][b]" +tempResultHolder[0] + "[/b][/size]" + "\n" + "[size=30][i]" + tempResultHolder[1] + "[/i][/size]" + "\n\n"
                     print(tempResultHolder[0], ' - ', tempResultHolder[1])
            except:
                pass
            print(resA)
        elif "Cannot access link" in resA:
            resA = "[b]The server was not able to retrieve the data from the input link[/b]"
        else:
            resultHolder = resA
            resA = "[b]The server was not able to retrieve any information regarding the input[/b]"

        resA = "[size=50][b]" + str(self.inputString) + "[/b][/size]" + "\n\n\n" +resA
        s.close()
        self.anim.repeat = False
        self.anim.stop(self.anim_bar)

        print("stopped widget")

        tempLayout = BoxLayout(orientation='vertical')
        counter=0
        empty = ""
        bracket = False
        for i in range(len(resA)):
            empty+=resA[i]
            if resA[i]=='[':
                bracket = True
                continue
            if resA[i]==']':
                bracket=False
                continue
            if bracket:
                continue
            if counter==40:
                empty+='\n'
                counter=0
                continue
            if resA[i]=='\n':
                counter=0
                continue
            counter+=1
        resA = empty
        result = Label(text=resA, markup=True)
        backButton = Button(text='Back')


        if "Verified" in resultHolder:
            color = [0,1,0,1]
        elif "Satiric" in resultHolder:
            color = [1,0,0,1]
        else:
            color = [1,1,0,1]

        #create pupup for displaying output
        popup = Popup(title=(resultHolder.split('\n'))[0],title_color = color,title_size=100,separator_color=color,title_align="center",auto_dismiss=True)
        top = ScrollView(size=(Window.width, Window.height))
        tempLayout.add_widget(result)
        tempLayout.add_widget(backButton)
        top.add_widget(tempLayout)
        popup.content = top
        backButton.bind(on_press = popup.dismiss)
        popup.open()
        self.ids.text_input.text = ""
        self.ids.submit_button.disable = False

    def hello(self, textChange):
        self.inputString = textChange

    def hello2(self, textChange):
        self.inputIP = textChange

class FactUApp(App):
    def build(self):
        return FactU()

if __name__ == '__main__':
    #run the main application
    FactUApp().run()
