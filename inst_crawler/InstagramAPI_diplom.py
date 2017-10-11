#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import random
import json
import hashlib
import hmac
import urllib
import uuid
import time
import sys
import re

#The urllib library was split into other modules from Python 2 to Python 3
if sys.version_info.major == 3:
    import urllib.parse

# from ImageUtils import getImageSize
from requests_toolbelt import MultipartEncoder
# from moviepy.editor import VideoFileClip

class InstagramAPI:
    API_URL = 'https://i.instagram.com/api/v1/'
    DEVICE_SETTINTS = {
    'manufacturer'      : 'Xiaomi',
    'model'             : 'HM 1SW',
    'android_version'   : 18,
    'android_release'   : '4.3'
    }
    USER_AGENT = 'Instagram 9.2.0 Android ({android_version}/{android_release}; 320dpi; 720x1280; {manufacturer}; {model}; armani; qcom; en_US)'.format(**DEVICE_SETTINTS)
    IG_SIG_KEY = '012a54f51c49aa8c5c322416ab1410909add32c966bbaa0fe3dc58ac43fd7ede'
    EXPERIMENTS = 'ig_android_progressive_jpeg,ig_creation_growth_holdout,ig_android_report_and_hide,ig_android_new_browser,ig_android_enable_share_to_whatsapp,ig_android_direct_drawing_in_quick_cam_universe,ig_android_huawei_app_badging,ig_android_universe_video_production,ig_android_asus_app_badging,ig_android_direct_plus_button,ig_android_ads_heatmap_overlay_universe,ig_android_http_stack_experiment_2016,ig_android_infinite_scrolling,ig_fbns_blocked,ig_android_white_out_universe,ig_android_full_people_card_in_user_list,ig_android_post_auto_retry_v7_21,ig_fbns_push,ig_android_feed_pill,ig_android_profile_link_iab,ig_explore_v3_us_holdout,ig_android_histogram_reporter,ig_android_anrwatchdog,ig_android_search_client_matching,ig_android_high_res_upload_2,ig_android_new_browser_pre_kitkat,ig_android_2fac,ig_android_grid_video_icon,ig_android_white_camera_universe,ig_android_disable_chroma_subsampling,ig_android_share_spinner,ig_android_explore_people_feed_icon,ig_explore_v3_android_universe,ig_android_media_favorites,ig_android_nux_holdout,ig_android_search_null_state,ig_android_react_native_notification_setting,ig_android_ads_indicator_change_universe,ig_android_video_loading_behavior,ig_android_black_camera_tab,liger_instagram_android_univ,ig_explore_v3_internal,ig_android_direct_emoji_picker,ig_android_prefetch_explore_delay_time,ig_android_business_insights_qe,ig_android_direct_media_size,ig_android_enable_client_share,ig_android_promoted_posts,ig_android_app_badging_holdout,ig_android_ads_cta_universe,ig_android_mini_inbox_2,ig_android_feed_reshare_button_nux,ig_android_boomerang_feed_attribution,ig_android_fbinvite_qe,ig_fbns_shared,ig_android_direct_full_width_media,ig_android_hscroll_profile_chaining,ig_android_feed_unit_footer,ig_android_media_tighten_space,ig_android_private_follow_request,ig_android_inline_gallery_backoff_hours_universe,ig_android_direct_thread_ui_rewrite,ig_android_rendering_controls,ig_android_ads_full_width_cta_universe,ig_video_max_duration_qe_preuniverse,ig_android_prefetch_explore_expire_time,ig_timestamp_public_test,ig_android_profile,ig_android_dv2_consistent_http_realtime_response,ig_android_enable_share_to_messenger,ig_explore_v3,ig_ranking_following,ig_android_pending_request_search_bar,ig_android_feed_ufi_redesign,ig_android_video_pause_logging_fix,ig_android_default_folder_to_camera,ig_android_video_stitching_7_23,ig_android_profanity_filter,ig_android_business_profile_qe,ig_android_search,ig_android_boomerang_entry,ig_android_inline_gallery_universe,ig_android_ads_overlay_design_universe,ig_android_options_app_invite,ig_android_view_count_decouple_likes_universe,ig_android_periodic_analytics_upload_v2,ig_android_feed_unit_hscroll_auto_advance,ig_peek_profile_photo_universe,ig_android_ads_holdout_universe,ig_android_prefetch_explore,ig_android_direct_bubble_icon,ig_video_use_sve_universe,ig_android_inline_gallery_no_backoff_on_launch_universe,ig_android_image_cache_multi_queue,ig_android_camera_nux,ig_android_immersive_viewer,ig_android_dense_feed_unit_cards,ig_android_sqlite_dev,ig_android_exoplayer,ig_android_add_to_last_post,ig_android_direct_public_threads,ig_android_prefetch_venue_in_composer,ig_android_bigger_share_button,ig_android_dv2_realtime_private_share,ig_android_non_square_first,ig_android_video_interleaved_v2,ig_android_follow_search_bar,ig_android_last_edits,ig_android_video_download_logging,ig_android_ads_loop_count_universe,ig_android_swipeable_filters_blacklist,ig_android_boomerang_layout_white_out_universe,ig_android_ads_carousel_multi_row_universe,ig_android_mentions_invite_v2,ig_android_direct_mention_qe,ig_android_following_follower_social_context'
    SIG_KEY_VERSION = '4'

    #username            # Instagram username
    #password            # Instagram password
    #debug               # Debug
    #uuid                # UUID
    #device_id           # Device ID
    #username_id         # Username ID
    #token               # _csrftoken
    #isLoggedIn          # Session status
    #rank_token          # Rank token
    #IGDataPath          # Data storage path

    def __init__(self, username, password, debug = False, IGDataPath = None):
        m = hashlib.md5()
        m.update(username.encode('utf-8') + password.encode('utf-8'))
        self.device_id = self.generateDeviceId(m.hexdigest())
        self.setUser(username, password)
        self.isLoggedIn = False
        self.LastResponse = None

    def setUser(self, username, password):
        self.username = username
        self.password = password
        self.uuid = self.generateUUID(True)

    def login(self, force = False):
        if (not self.isLoggedIn or force):
            self.s = requests.Session()
            # if you need proxy make something like this:
            # self.s.proxies = {"https" : "http://proxyip:proxyport"}
            if (self.SendRequest('si/fetch_headers/?challenge_type=signup&guid=' + self.generateUUID(False), None, True)):

                data = {'phone_id'   : self.generateUUID(True),
                        '_csrftoken' : self.LastResponse.cookies['csrftoken'],
                        'username'   : self.username,
                        'guid'       : self.uuid,
                        'device_id'  : self.device_id,
                        'password'   : self.password,
                        'login_attempt_count' : '0'}

                if (self.SendRequest('accounts/login/', self.generateSignature(json.dumps(data)), True)):
                    self.isLoggedIn = True
                    self.username_id = self.LastJson["logged_in_user"]["pk"]
                    self.rank_token = "%s_%s" % (self.username_id, self.uuid)
                    self.token = self.LastResponse.cookies["csrftoken"]

                    self.syncFeatures()
                    print ("Login success!\n")
                    return True;

    def syncFeatures(self):
        data = json.dumps({
        '_uuid'         : self.uuid,
        '_uid'          : self.username_id,
        'id'            : self.username_id,
        '_csrftoken'    : self.token,
        'experiments'   : self.EXPERIMENTS
        })
        return self.SendRequest('qe/sync/', self.generateSignature(data))

    def megaphoneLog(self):
        return self.SendRequest('megaphone/log/')

    def expose(self):
        data = json.dumps({
        '_uuid'        : self.uuid,
        '_uid'         : self.username_id,
        'id'           : self.username_id,
        '_csrftoken'   : self.token,
        'experiment'   : 'ig_android_profile_contextual_feed'
        })
        return self.SendRequest('qe/expose/', self.generateSignature(data))

    def logout(self):
        logout = self.SendRequest('accounts/logout/')

    def generateSignature(self, data):
        try:
            parsedData = urllib.parse.quote(data)
        except AttributeError:
            parsedData = urllib.quote(data)

        return 'ig_sig_key_version=' + self.SIG_KEY_VERSION + '&signed_body=' + hmac.new(self.IG_SIG_KEY.encode('utf-8'), data.encode('utf-8'), hashlib.sha256).hexdigest() + '.' + parsedData

    def generateDeviceId(self, seed):
        volatile_seed = "12345"
        m = hashlib.md5()
        m.update(seed.encode('utf-8') + volatile_seed.encode('utf-8'))
        return 'android-' + m.hexdigest()[:16]

    def generateUUID(self, type):
        #according to https://github.com/LevPasha/Instagram-API-python/pull/16/files#r77118894
        #uuid = '%04x%04x-%04x-%04x-%04x-%04x%04x%04x' % (random.randint(0, 0xffff),
        #    random.randint(0, 0xffff), random.randint(0, 0xffff),
        #    random.randint(0, 0x0fff) | 0x4000,
        #    random.randint(0, 0x3fff) | 0x8000,
        #    random.randint(0, 0xffff), random.randint(0, 0xffff),
        #    random.randint(0, 0xffff))
        generated_uuid = str(uuid.uuid4())
        if (type):
            return generated_uuid
        else:
            return generated_uuid.replace('-', '')

    def SendRequest(self, endpoint, post = None, login = False):
        # print(endpoint)
        if (not self.isLoggedIn and not login):
            raise Exception("Not logged in!\n")
            return;

        self.s.headers.update ({'Connection' : 'close',
                                'Accept' : '*/*',
                                'Content-type' : 'application/x-www-form-urlencoded; charset=UTF-8',
                                'Cookie2' : '$Version=1',
                                'Accept-Language' : 'en-US',
                                'User-Agent' : self.USER_AGENT})

        if (post != None): # POST
            response = self.s.post(self.API_URL + endpoint, data=post) # , verify=False
        else: # GET
            response = self.s.get(self.API_URL + endpoint) # , verify=False

        if response.status_code == 200:
            self.LastResponse = response
            self.LastJson = json.loads(response.text)
            return True
        else:
            print ("Request return " + str(response.status_code) + " error!")
            # for debugging
            try:
                self.LastResponse = response
                self.LastJson = json.loads(response.text)
            except:
                pass
            return False

    def configure(self, upload_id, photo, caption = ''):
        (w,h) = getImageSize(photo)
        data = json.dumps({
        '_csrftoken'    : self.token,
        'media_folder'  : 'Instagram',
        'source_type'   : 4,
        '_uid'          : self.username_id,
        '_uuid'         : self.uuid,
        'caption'       : caption,
        'upload_id'     : upload_id,
        'device'        : self.DEVICE_SETTINTS,
        'edits'         : {
            'crop_original_size': [w * 1.0, h * 1.0],
            'crop_center'       : [0.0, 0.0],
            'crop_zoom'         : 1.0
        },
        'extra'         : {
            'source_width'  : w,
            'source_height' : h,
        }})
        return self.SendRequest('media/configure/?', self.generateSignature(data))

    def getUsernameInfo(self, usernameId):
        return self.SendRequest('users/'+ str(usernameId) +'/info/')

    def getUserFollowings(self, usernameId, maxid = ''):
        return self.SendRequest('friendships/'+ str(usernameId) +'/following/?'
            + 'ig_sig_key_version='+ self.SIG_KEY_VERSION
            + '&rank_token='+ self.rank_token
            + ('&max_id='+ str(maxid) if maxid else ''))

    def getUserFollowers(self, usernameId, maxid = ''):
        return self.SendRequest('friendships/'+ str(usernameId) +'/followers/?'
            +'rank_token='+ self.rank_token
            + '&max_id='+ str(maxid) if maxid else '')
            
    def getTotalFollowers(self,usernameId):
        followers = []
        next_max_id = ''
        while 1:
            self.getUserFollowers(usernameId,next_max_id)
            temp = self.LastJson

            for item in temp["users"]:
                followers.append(item)

            if temp["big_list"] == False:
                return followers            
            next_max_id = temp["next_max_id"]         

    def getTotalFollowings(self, usernameId):
        followers = []
        next_max_id = 0
        error_codes = (400, 429, 404)
        while 1:
            self.getUserFollowings(usernameId, next_max_id)
            if self.LastResponse.status_code in error_codes:
                return None
            temp = self.LastJson
            followers += temp["users"]

            if not temp['users'] or not temp.get('big_list', True) or type(temp.get('next_max_id')) != int:
                return followers            
            next_max_id += 200
    
    def __is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def getId(self, username):
        link = 'http://www.instagram.com/' + username
        content = requests.get(link).text
        try:
            start_index = (content.index('"owner": {"id": "')) + len('"owner": {"id": "')
        except ValueError:
            return None

        test_string = ''
        for collect in range(14):
            test_string = test_string + content[start_index]
            start_index = start_index + 1

        edit_string = ''

        for char in test_string:
            if self.__is_number(char) == True:
                edit_string = edit_string + char
            else:
                edit_string = edit_string + ''
        return edit_string

    def getPrivId(self, uname):
        url = "https://smashballoon.com/instagram-feed/find-instagram-user-id/?username=%s" % uname    
        html = requests.get(url).text 
        s = re.search('User ID: (\d+)', html)
        return s.group(1) if s else None