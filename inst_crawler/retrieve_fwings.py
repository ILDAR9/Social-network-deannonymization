import sys, time
from utils import DB, get_api
import re, string

credentials = ({
        ('login1','pass1'),
        ('login2','pass2'),
        ('login3','pass3'),
        ('login4','pass4'),
        ('login5','pass5'),
        ('login6','pass6'),
        })


pattern = re.compile('[^a-zа-я]+')
STOPWORDS = """
design architecture архитектура remont kazan украшения мастерская мебель ручной работы работа
стрижка казани дизайнер одежда studio кальянная shisha цветы flowers салон рестора кафе официальный
журнал journal декор мебель вещи униформа show отель хостел hostel hotel продвигаи реклама волонтеры клуб
химчистка часы телефон кальяна квартиры казань kazan доступ продажи продажа стоматология cafe restaurant
nk нк нижнекамк нижнекамска альметьевск альметьевска ногти маникюр макияж заказ секс sex porn трусы дома
barber smoker smoke авто торты торт
""".split()

def filter_advert(fwings):
    filt = filter(lambda item: not any(w in STOPWORDS for w in pattern.split(item['full_name'].lower())), fwings)
    res = []
    for item in filt:
        item['full_name'] = item['full_name'][:40]
        res.append(item)
    return res
 
def set_proceed(db, pk):
    db.execute("UPDATE uids SET is_proceed = TRUE WHERE pk = %d" % pk)

def proceed_followings(db, fwngs, pk):
    args_fwings = db.prep_arg(db.cur.mogrify( "(%s,%s)", (min(pk, fw['pk']), max(pk, fw['pk'])))
                                   for fw in fwngs)
    db.execute("INSERT INTO fwings (lpk, rpk) VALUES " + args_fwings + " ON CONFLICT DO NOTHING")
    
    args_personal = db.prep_arg(db.cur.mogrify( "(%s,%s,%s)", 
                                    (fw.get('pk'), fw.get('full_name'), fw.get('username')))
                                    for fw in fwngs)
    db.execute("INSERT INTO personal (pk, fname, uname) VALUES " + args_personal + " ON CONFLICT DO NOTHING")


def iter_uids(limit, ind_cred):
    API_BLOCKED_STATUS = 429
    API_BAD_REQUEST = 400
    NOT_FOUND_STATUS = 404
    gen_api = get_api(credentials[ind_cred])
    api = next(gen_api)

    def get_fwings(pk):
        try:
            return api.getTotalFollowings(pk)
        except:
            api.logout()
            api.login()
            return api.getTotalFollowings(pk)

    with DB('instadata') as idb:
        uids = idb.execute_get('SELECT pk FROM uids where is_proceed=FALSE AND is_private=FALSE LIMIT %d OFFSET %d' % (limit, limit*ind_cred))
        print(uids)
        for i, urow in enumerate(uids):
            pk = urow[0]
            fwngs = get_fwings(pk)
            if not fwngs or len(fwngs) == 0:
                print("Followings Empty", pk)
                if api.LastResponse.status_code == NOT_FOUND_STATUS:
                    print('NOT_FOUND fwings %s' % pk)
                    continue
                while api.LastResponse.status_code == API_BLOCKED_STATUS or api.LastResponse.status_code == API_BAD_REQUEST:
                    api.logout()
                    if attempt > 2:
                        print("API stop working on pk=%s" % pk)
                        time.sleep(60)
                    api = next(gen_api)
                    print('Change instagram user')
                    attempt += 1
                    fwngs = get_fwings(pk)

            attempt = 0

            print('%d %s -> len=%d' % (i, pk, len(fwngs)))
            try:
                if fwngs:
                    fwngs_filt = filter_advert(fwngs)
                    if fwngs_filt:
                        proceed_followings(idb, fwngs_filt, pk)
                set_proceed(idb, pk)    
            except:
                print('pk = ', pk)
                raise
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Start from index needed as arg")
        sys.exit(1)
    ind_cred = int(sys.argv[1])
    if ind_cred < 0 or ind_cred >1:
        print("Cred index is 0 or 1")
        sys.exit(1)
    iter_uids(int(sys.argv[2]), ind_cred)