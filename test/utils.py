import os
import json


def move(client, x, y, theta):
    print(f"Moving to (x, y, theta)=({x}, {y}, {theta}) ...")
    client.move_to_pose(x, y, theta)
    # 待たないとis_command_runinngが反映されない 実行後すぐはFalseのままになっている
    # time.sleep(1)
    # cnt = 0
    # while client.is_command_running(): # 今実行中かどうか
    #     time.sleep(1)
    #     cnt += 1
    #     print(1)
    #     if cnt == 10:
    #         client.cancel_command()
    #         break
    #         print("10 seconds have elapsed.")
    # print(f"cnt: {cnt}")
    result = client.get_last_command_result()[0]
    if result.success:
        print("Success!")
    else:
        with open(f"/home/{os.environ['USER']}/kachaka-api/docs/KachakaErrorCode.json") as f:
            error_codes = json.load(f)
        for error_code in error_codes:
            if int(error_code["code"]) == result.error_code:
                error_title = error_code["title"]
                error_description = error_code["description"]
                print(f"Failure: {error_title}\n{error_description}") 