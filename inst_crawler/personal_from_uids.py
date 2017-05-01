import sys, time
from utils import DB, get_api

credentials = {
#add here some users
        ('login1','pass1'),
        ('login2','pass2'),
        ('login3','pass3'),
        ('login4','pass4'),
        ('login5','pass5'),
        ('login6','pass6'),
        }

def init_personal_from_file(start_from):  
    API_BLOCKED_STATUS = 429
    API_BAD_REQUEST = 400
    NOT_FOUND_STATUS = 404
    with open('inst_uids_init.csv', 'r') as f:
        lines = f.readlines()
    gen_api = get_api(credentials)
    api = next(gen_api)
    attempt = 0
    with DB('instadata') as idb:
        for i, pk, _, uname in (l.strip().split() for l in lines[start_from:]):
            if not pk.isdigit():
                print('%s FAIL %s %s' % (i, pk, uname))
                continue
            if not api.getUsernameInfo(pk):    
                if api.LastResponse.status_code == NOT_FOUND_STATUS:
                    print('%s NOT_FOUND %s %s' % (i, pk, uname))
                    continue
                while api.LastResponse.status_code == API_BLOCKED_STATUS or api.LastResponse.status_code == API_BAD_REQUEST:
                    api.logout()
                    if attempt > 5:
                        print("API stop working on pk=%s uname=%s (%s)" % (pk, attempt, uname))
                        time.sleep(60)
                    api = next(gen_api)
                    print('Change instagram user')
                    attempt += 1
                    api.getUsernameInfo(pk)
                
            attemp = 0 
                
            fname = api.LastJson['user'].get('full_name', '').strip()
            fname = fname if fname else None
            
            print('%s %s' % (i, pk))
            
            idb.execute("INSERT INTO personal (pk, uname, fname) VALUES "
                        + idb.cur.mogrify("(%s, %s, %s)", (pk, uname, fname)).decode()
                        + "ON CONFLICT DO NOTHING")
            time.sleep(0.2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Start from index needed as arg")
        sys.exit(1)
    init_personal_from_file(int(sys.argv[1]))
