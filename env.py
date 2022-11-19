import numpy as np
def initEnv():

    env_state = np.full(206,0)
    env_state[:9] = 32
    env_state[9] = 128
    env_state[10] = 20000
    env_state[11:17] = 1000

    for a_ in range(7):
        env_state[29 + a_*2] = 1

    return env_state


def getAgentState(env_state):

    P_state = np.full(158,0) 
    P_id = int((env_state[43]%14)//2)
    P1_id = int(env_state[43]%14)
    for i in range(7):
        if (P_id+i) <= 6:
            P_state[i] = env_state[10+P_id+i]
        elif (P_id+i) > 6:
            P_state[i] = env_state[3+P_id+i]
    for lct in range(7):
        if (P_id+lct) <= 6:
            P_state[(7+20*lct):(7+20*(lct+1))] = env_state[(44+(P_id+lct)*22):(42+(P_id+lct+1)*22)]
        elif (P_id+lct) > 6:
            P_state[(7+20*lct):(7+20*(lct+1))] = env_state[(44+(P_id+lct-7)*22):(42+(P_id+lct-6)*22)]

    if P1_id <=1:
        P_state[147] = 1
    if P1_id >= 2:
        P_state[147] = env_state[15+P1_id]
    P_state[148] = env_state[198]
    for dtb in range(7):
        if (P_id+i) <= 6:
            P_state[149+dtb] = env_state[199+P_id+dtb]
        elif (P_id+1) > 6:
            P_state[149+dtb] = env_state[192+P_id+dtb]
    P_state[157] = P_id
    return P_state.astype(np.float64)


def getValidActions(P_state):
    
    Valid_Actions_return = np.full(9,0)
    Check_place_a_bet = P_state[147]
    Check_coin_player = P_state[:7]
    Card_on_hand = P_state[7:27]
    Card_on_hand_1 = P_state[7:17]
    Card_on_hand_2 = P_state[17:27]
    Sum_number_of_card = np.sum(Card_on_hand)

    if Check_place_a_bet == 0:
        if Check_coin_player[0] >= 100:
            Valid_Actions_return[0:4] = 1
        elif Check_coin_player[0]<100 and Check_coin_player[0]>=25:
            Valid_Actions_return[0:3] = 1
            Valid_Actions_return[3] = 0
        elif Check_coin_player[0]<25 and Check_coin_player[0]>=10:
            Valid_Actions_return[0:2] = 1
            Valid_Actions_return[2:4] = 0
        elif Check_coin_player[0]<10 and Check_coin_player[0]>=5:
            Valid_Actions_return[0] = 1
            Valid_Actions_return[1:4] = 0
        elif Check_coin_player[0]<5:
            Valid_Actions_return[4] = 1
    elif Check_place_a_bet == 1:
        Valid_Actions_return[8] = 1                   
    elif Check_place_a_bet != 0 and Check_place_a_bet != 1:
        check_place = 0
        for s_ in range(len(Card_on_hand_2)):
            if Card_on_hand_2[s_] != 0:
                check_place += 1

        card_other_0 = 0
        for run in range(len(Card_on_hand)):
            if Card_on_hand[run] != 0:
                card_other_0 += 1

        if card_other_0==1 and Sum_number_of_card==2:
            if check_place == 0:
                if Check_coin_player[0] >= Check_place_a_bet:
                    Valid_Actions_return[4:8] = 1
        if card_other_0!=1 and Sum_number_of_card==2:
            if check_place == 0:
                if Check_coin_player[0] >= Check_place_a_bet:
                    Valid_Actions_return[4:7] = 1
        if Sum_number_of_card >= 3 and check_place == 0:
            Valid_Actions_return[4:6] = 1
        if check_place != 0:
            Valid_Actions_return[4:6] = 1
    return Valid_Actions_return.astype(np.int64)




def stepEnv(action,env_state):
    
    P_player = int((env_state[43]%14)//2)
    status_player = env_state[29:43]
    card_on_hand = env_state[(44+P_player*22):(32+(P_player+1)*22)] 
    card_on_hand_2 = env_state[(32+(P_player+1)*22):(42+(P_player+1)*22)]
    point_card = env_state[42+(P_player+1)*22]
    point_card_2 = env_state[43+(P_player+1)*22]
    remaining = np.sum(env_state[:10])
    if remaining == 0:
        env_state[:10] = [32,32,32,32,32,32,32,32,32,128]

    if status_player[int(env_state[43]%14)] == 0:
        env_state[43] += 1
    if status_player[int(env_state[43]%14)] == 1: 
        
        if P_player >= 1: 
            if action == 0:
                env_state[15+2*P_player] += 5 #tiền đặt
                env_state[10+P_player] -= 5   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        point_card = point_card + 11

                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card
                 
                env_state[43] += 1
            if action == 1:
                env_state[15+2*P_player] += 10 #tiền đặt
                env_state[10+P_player] -= 10   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        point_card = point_card + 11
                            
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card

                env_state[43] += 1

            if action == 2:
                env_state[15+2*P_player] += 25 #tiền đặt
                env_state[10+P_player] -= 25   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        point_card = point_card + 11
                            
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card

                env_state[43] += 1
            if action == 3:
                env_state[15+2*P_player] += 100 #tiền đặt
                env_state[10+P_player] -= 100   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        point_card = point_card + 11
                            
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card

                env_state[43] += 1
            if action == 4:
                status_player[int(env_state[43]%14)] == 0
                env_state[29+(env_state[43]%14)] = 0
                
                env_state[43] += 1
            if action == 5:
                if env_state[43]%2 == 0:
                    weighted_random = np.array(env_state[:10])
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1

                    check_point_other_A = np.sum(card_on_hand[1:10])
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        if check_point_other_A <= 10:
                            point_card += 11
                        elif check_point_other_A >= 11:
                            point_card += 1      
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card

                    env_state[43] += 1
                if env_state[43]%2 != 0:
                    weighted_random = np.array(env_state[:10])
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand_2[choice_place] += 1

                    check_point_other_A_1 = np.sum(card_on_hand_2[1:10])
                    if choice_place >= 1:
                        point_card_2 = point_card_2 + choice_place + 1
                    if choice_place == 0:
                        if check_point_other_A_1 <= 10:
                            point_card_2 += 11
                        elif check_point_other_A_1 >= 11:
                            point_card_2 += 1      
                    env_state[(32+(P_player+1)*22):(42+(P_player+1)*22)] = card_on_hand_2
                    env_state[43+(P_player+1)*22] = point_card

                    env_state[43] += 1
            if action == 6:
                env_state[15+2*P_player] *= 2
                env_state[10+P_player] -= (env_state[15+2*P_player]/2)

                weighted_random = np.array(env_state[:10])
                rate_random = weighted_random/np.sum(weighted_random)
                choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                weighted_random[choice_place] -= 1
                card_on_hand[choice_place] += 1
                
                check_point_other_A = np.sum(card_on_hand[1:10])
                if choice_place >= 1:
                    point_card = point_card + choice_place + 1
                if choice_place == 0:
                    if check_point_other_A <= 10:
                        point_card += 11
                    elif check_point_other_A >= 11:
                        point_card += 1      
                env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                env_state[42+(P_player+1)*22] = point_card

                env_state[43] += 1             
            if action == 7:
                env_state[16+2*P_player] = env_state[15+2*P_player]
                env_state[10+P_player]-= env_state[16+2*P_player]
                env_state[30+(env_state[43]%14)] = 1

                card_split = env_state[(44+P_player*22):(32+(P_player+1)*22)]
                for s_ in range(len(card_split)):
                    if card_split[s_] == 2:
                        card_split[s_] -= 1
                        env_state[32+(P_player+1)*22+s_] += 1
                        env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_split
                if card_split[0] == 1:
                    env_state[42+(P_player+1)*22] = 11
                    env_state[43+(P_player+1)*22] = 11
                if card_split[0] == 0:
                    env_state[42+(P_player+1)*22] = point_card/2
                    env_state[42+(P_player+1)*22] = env_state[42+(P_player+1)*22] 

                env_state[43] += 2
        if P_player == 0:
            if action == 8:
                status_bos = int(env_state[43]%14)
                group_one_on_board = 0
                for i_s in range(len(card_on_hand)):
                    if card_on_hand[i_s] != 0:
                        group_one_on_board += 1
                asgroup_1 = np.sum(card_on_hand)
                asgroup_2 = np.sum(card_on_hand_2)

                if asgroup_1 == 0 and asgroup_2 == 0:
                    weighted_random = np.array(env_state[:10])
                    for i_ in range(2):
                        rate_random = weighted_random/np.sum(weighted_random)
                        choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                        weighted_random[choice_place] -= 1
                        card_on_hand[choice_place] += 1
                        if choice_place >= 1:
                            point_card = point_card + choice_place + 1
                        if choice_place == 0:
                            point_card = point_card + 11
                                
                    env_state[44:54] = card_on_hand
                    env_state[64] = point_card

                    env_state[43] += 1

                if group_one_on_board == 1 and asgroup_1 == 2:  #tách bài
                    if asgroup_2 == 0:
                        card_split = env_state[44:54]
                        for s_ in range(len(card_split)):
                            if card_split[s_] == 2:
                                card_split[s_] -= 1
                                env_state[54+s_] += 1
                                env_state[44:54] = card_split
                        if card_split[0] == 1:
                            env_state[64] = 11
                            env_state[65] = 11
                        if card_split[0] == 0:
                            env_state[64] = point_card/2
                            env_state[65] = env_state[64] 
                        env_state[43] += 2
                elif env_state[64+status_bos] > 11:   #ko rút bài nữa
                    status_player[status_bos] = 0
                    env_state[29+status_bos] = 0

                    env_state[43] += 1
                elif (env_state[64+status_bos]<=11) and (env_state[64+status_bos]>0):   # rút thêm 1 lá
                    decks_of_card = env_state[(44+status_bos*10):(44+(status_bos+1)*10)]
                    weighted_random = np.array(env_state[:10])
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)

                    weighted_random[choice_place] -= 1
                    decks_of_card[choice_place] += 1

                    check_point_other_A1 = np.sum(decks_of_card[1:10])
                    if choice_place >= 1:
                        env_state[64+status_bos] = env_state[64+status_bos] + choice_place + 1
                    if choice_place == 0:
                        if check_point_other_A1 <= 10:
                            env_state[64+status_bos] += 11
                        elif check_point_other_A1 >= 11:
                            env_state[64+status_bos] += 1      
                    env_state[(44+status_bos*10):(44+(status_bos+1)*10)] = decks_of_card

                    env_state[43] += 1  
                      
                
    #-------------------------------------#####reset_small_game_---------------------------------------#

    point_end = env_state[np.array([64,65,86,87,108,109,130,131,152,153,174,175,196,197])]
    for zes in range(14):
        if point_end[zes] >= 21:
            status_player[zes] = 0
    check_small_game = np.sum(status_player)
    if check_small_game == 0:
        cardNumbers = []
        for sz_ in range(7):
            cardNumbers.append(np.sum(env_state[(44+sz_*22):(42+(sz_+1)*22)]))
        check_2 = [] #----------
        for isd in range(7):
            check_2.append(point_end[2*isd] + cardNumbers[isd])
        check_blackjack = np.array(check_2)
        blackjackPlaces = np.where(check_blackjack == 23)[0]
        asd = [0,1,2,3,4,5,6]
        for rub in blackjackPlaces:
            asd.remove(rub)
        if len(blackjackPlaces) != 0:
            if blackjackPlaces[0] == 0:
                for run_ in blackjackPlaces[1:]:
                    env_state[10+run_] += env_state[15+2*run_]
                    env_state[15+2*run_] = 0
                for rub_ in asd:
                    env_state[10] += env_state[15+2*rub_] + env_state[16+2*rub_]
                    env_state[15+2*rub_] = 0
                    env_state[16+2*rub_] = 0   
                    env_state[199] += 1
            if blackjackPlaces[0] != 0:
                for runn in blackjackPlaces:
                    env_state[10] -= env_state[15+2*runn]
                    env_state[10+runn] += 2*env_state[15+2*runn]
                    env_state[15+2*runn] = 0
                    env_state[199+runn] += 1 
                
                asd.remove(0)
                for ez in range(2):
                    for dct in asd:
                        if point_end[2*dct] >= 22:
                            env_state[10] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199] += 1
                        if point_end[2*dct+1] >= 22:
                            env_state[10] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            env_state[199] += 1
                        if point_end[2*dct] <= 21:
                            if point_end[2*dct] == point_end[ez]:
                                env_state[10+dct] += env_state[15+2*dct]
                                env_state[15+2*dct] = 0
                            elif point_end[2*dct] > point_end[ez]:
                                env_state[10+dct] += 2*env_state[15+2*dct]
                                env_state[10] -= env_state[15+2*dct]
                                env_state[15+2*dct] = 0
                                env_state[199+dct] += 1
                            elif point_end[2*dct] < point_end[ez]:
                                env_state[10] += env_state[15+2*dct]
                                env_state[15+2*dct] = 0
                                env_state[199] += 1
                        if point_end[2*dct+1] <= 21:
                            if point_end[2*dct+1] == point_end[ez]:
                                env_state[10+dct] += env_state[16+2*dct]
                                env_state[16+2*dct] = 0
                            elif point_end[2*dct+1] > point_end[ez]:
                                env_state[10+dct] += 2*env_state[16+2*dct]
                                env_state[10] -= env_state[16+2*dct]
                                env_state[16+2*dct] = 0
                                env_state[199+dct] += 1
                            elif point_end[2*dct+1] < point_end[ez]:
                                env_state[10] += env_state[16+2*dct]
                                env_state[16+2*dct] = 0
                                env_state[199] += 1
        else:
            for ez in range(2):
                asd = [1,2,3,4,5,6]
                for dct in asd:
                    if point_end[2*dct] >= 22:
                        env_state[10] += env_state[15+2*dct]
                        env_state[15+2*dct] = 0
                        env_state[199] += 1
                    if point_end[2*dct+1] >= 22:
                        env_state[10] += env_state[16+2*dct]
                        env_state[16+2*dct] = 0
                        env_state[199] += 1
                    if point_end[2*dct] <= 21:
                        if point_end[2*dct] == point_end[ez]:
                            env_state[10+dct] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                        elif point_end[2*dct] > point_end[ez]:
                            env_state[10+dct] += 2*env_state[15+2*dct]
                            env_state[10] -= env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199+dct] += 1
                        elif point_end[2*dct] < point_end[ez]:
                            env_state[10] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199] += 1
                    if point_end[2*dct+1] <= 21:
                        if point_end[2*dct+1] == point_end[ez]:
                            env_state[10+dct] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                        elif point_end[2*dct+1] > point_end[ez]:
                            env_state[10+dct] += 2*env_state[16+2*dct]
                            env_state[10] -= env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            env_state[199+dct] += 1
                        elif point_end[2*dct+1] < point_end[ez]:
                            env_state[10] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            env_state[199] += 1

        env_state[43:198] = 0        
        for a_s in range(7):
            env_state[29 + a_s*2] = 1
        env_state[198] += 1
    return env_state



def getAgentsize():
    return 7

def checkEnded(env_state):
    pointArr = env_state[10:17]
    if env_state[198] == 50:
        pointArr[0] -= 20000
        for edv in range(6):
            pointArr[edv+1] -= 1000
        maxPoint = np.max(pointArr)
        maxPointPlay = np.where(pointArr==maxPoint)[0]
        if len(maxPointPlay) == 1:
            return maxPointPlay[0] 
        else:
            number_of_win_smallgame = env_state[199:206]
            maxWin_smallgame = np.max(number_of_win_smallgame)
            maxWin_smallgame_Play = np.where(number_of_win_smallgame == maxWin_smallgame)[0]
            
            return maxWin_smallgame_Play[0]
    else:
        AF_end = 0
        for tex in range(7):
            if pointArr[tex] >= 5:
                AF_end += 1
        if AF_end == 1:
            winpoint = np.where(pointArr >= 5)[0]
            return winpoint
        else:
            return -1       



def getReward(P_state):
    if P_state[148] != 50:
        scorePoint_Arr = P_state[0:7]
        money_left = 0
        for ted in range(7):
            if scorePoint_Arr[ted] >= 5:
                money_left += 1
        if money_left == 1:
            winnerx = np.where(scorePoint_Arr >= 5)[0]
            if winnerx == 0:
                return 1
            else:
                return -1
            

        return 0
    else:
        scorePoint_Arr = P_state[0:7]
        maxCoin_pl = np.max(scorePoint_Arr)
        if scorePoint_Arr[0] < maxCoin_pl:
            return -1
        else:
            maxCoin_Pl_place = np.where(scorePoint_Arr==maxCoin_pl)
            if len(maxCoin_Pl_place) == 1:
                return 1
            else:
                maxNumber_ofSmallWin = P_state[149:156]
                maxWin = np.max(maxNumber_ofSmallWin)
                if maxNumber_ofSmallWin[0] < maxWin:
                    return -1
                else:    #trường hợp nếu số ván thắng nhỏ bằng max
                    maxWin_player = np.where(maxNumber_ofSmallWin==maxWin)   #những người có số ván thắng nhỏ giống nhau
                if len(maxWin_player) == 1:
                    return 1
                else:
                    add = np.full(len(maxWin_player, P_state[157]))
                    setPlayer = maxWin_player + add
                    for place in range(len(setPlayer)):
                        if setPlayer[place] > 6:
                            setPlayer[place] -= 7
                    winner = np.min(setPlayer)
                    if setPlayer[0] == winner:
                        return 1
                    else:
                        return -1


def getStateSize():
   return 158

def run(listAgent,perData):
    env_state = initEnv()
    tempData = [[0],[0],[0],[0],[0],[0],[0]]

    winner = -1
    Id_player = int((env_state[43]%4)//2)
    while env_state[198] <= 50:
        pIdx = int((env_state[43]%4)//2)
        P1_state = getAgentState(env_state)
        list_action = getValidActions(P1_state)
        try:
            action, tempData[pIdx], perData = listAgent[pIdx](P1_state, tempData[pIdx], perData)
        except:
            print(list(env_state))

        if list_action[action] != 1:
            raise Exception('Người chơi trả về action lỗi') 

        stepEnv(action,env_state)
        winner = checkEnded(env_state)
        if winner != -1:
            break
    
    for pIdx in range(7):
        Id_player = pIdx
        P1_state = getAgentState(env_state)
        action, tempData[pIdx], perData = listAgent[pIdx](P1_state, tempData[pIdx], perData)


    return winner, perData


def main(listAgent, num_math, perData):
    if len(listAgent) != 7:
        print('Hệ thống cho đúng 7 người chơi:>>>>')

    numWin = np.full(8,0)
    pIdOrder = np.arange(7)
    for w_ in range(num_math):
        np.random.shuffle(pIdOrder)
        winner, perData = run([listAgent[pIdOrder[0]], listAgent[pIdOrder[1]], listAgent[pIdOrder[2]], listAgent[pIdOrder[3]], listAgent[pIdOrder[4]], listAgent[pIdOrder[5]], listAgent[pIdOrder[6]]], perData)

        if winner == -1:
            numWin[7] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    
    return numWin, perData


def random_player(P_state, tempData, perData):
    actions = getValidActions(P_state)
    actions = np.where(actions == 1)[0]
    # print(actions)
    action = np.random.choice(actions)
    return action, tempData, perData

win, _ = main([random_player]*getAgentsize(), 10000, [0])
print(win)