#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np

REDUNDANT_IOU_THRESH = 0.8
CROSSING_IOU_THRESH = 0.3
TRACKING_IOU_THRESH = 0.2
MAX_ABSENT_FRAME = 10
labels_to_names = {-1: 'nohat', 1: 'hat'}


def iou(box1, box2):
    bbox1 = [float(x) for x in box1]
    bbox2 = [float(x) for x in box2]

    (left_1, top_1, right_1, bottom_1) = bbox1
    (left_2, top_2, right_2, bottom_2) = bbox2

    left = max(left_1, left_2)
    top = max(top_1, top_2)
    right = min(right_1, right_2)
    bottom = min(bottom_1, bottom_2)

    # check if there is an overlap
    if right - left <= 0 or bottom - top <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (right_1 - left_1) * (bottom_1 - top_1)
    size_2 = (right_2 - left_2) * (bottom_2 - top_2)
    size_intersection = (right - left) * (bottom - top)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union


def pick_obj_by_id(tracker, index):
    for obj in tracker:
        if obj["id"] == index:
            return obj


def distance_between_boxes(box1, box2):
    p1 = [(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0]
    p2 = [(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0]
    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return d


class Tracker(object):
    def __init__(self):
        self.keys = ["unique_id", "id", "frame", "position", "class", "score", "pred_cross", "n_absence",
                     "pred_position", "track_score", "new_one", "n_occurrence", "show", "crossing",
                     "contains"]
        self.tracker = []
        self.tracker_record = []
        self.init_thresh = 0.8
        self.update_thresh = 0.4
        self.latest_id = 1
        self.unique_id = 1
        self.cross_zone = 1

    def predict_position(self):
        pass

    def find_latest_id(self):
        latest_id = 1
        for obj in self.tracker[1:]:
            if obj["id"] > latest_id:
                latest_id = obj["id"]
        latest_id += 1
        return latest_id

    def get_recorded_obj_by_id(self, index, frame):
        for obj in self.tracker_record[frame - 1][1:]:
            if obj.get("id") == index:
                return obj

    def moving_average_score(self, index, period):
        # Calculate the moving average score for past n period
        pass

    def track_score(self, index, frame, period=6):
        score = 0
        for i in range(period):
            idx = i + 1
            old_frame = frame - idx
            if old_frame > 0:
                old_obj = self.get_recorded_obj_by_id(index, old_frame)
                if old_obj is not None:
                    score += old_obj["score"] * old_obj["class"]
                else:
                    continue
            else:
                break
        return score

    def drop_redundant_obj_by_iou(self):
        drop_ids = []
        tracker_copy = []
        for i, obj1 in enumerate(self.tracker):
            if obj1["unique_id"] in drop_ids:
                continue
            bbox1 = obj1["position"]
            for obj2 in self.tracker[i + 1:]:
                if obj2["unique_id"] in drop_ids:
                    continue
                bbox2 = obj2["position"]
                if iou(bbox1, bbox2) >= REDUNDANT_IOU_THRESH:
                    print("redundant_iou:", iou(bbox1, bbox2))
                    if abs(obj1["score"][0]) > abs(obj2["score"][0]):
                        obj1["score"][0] = obj1["score"][0] / (abs(obj1["score"][0]) + abs(obj2["score"][0]))
                        drop_ids.append(obj2["unique_id"])
                    else:
                        obj2["score"][0] = obj2["score"][0] / (abs(obj1["score"][0]) + abs(obj2["score"][0]))
                        drop_ids.append(obj1["unique_id"])
        for i, obj in enumerate(self.tracker):
            if obj["unique_id"] not in drop_ids:
                tracker_copy.append(obj)
        self.tracker = tracker_copy

    def drop_redundant_tracker_by_id(self):
        drop_ids = []
        for i, obj1 in enumerate(self.tracker[1:]):
            i += 1
            if obj1["unique_id"] in drop_ids:
                continue
            for obj2 in self.tracker[i + 1:]:
                if obj2["unique_id"] in drop_ids:
                    continue
                if obj1["id"] == obj2["id"]:
                    if obj1["score"] > obj2["score"]:
                        drop_ids.append(obj2["unique_id"])
                    else:
                        drop_ids.append(obj1["unique_id"])
        for i, obj in enumerate(self.tracker[1:]):
            i += 1
            if obj["unique_id"] in drop_ids:
                del self.tracker[i]

    def match_after_crossing(self, after_crossing_objs, before_crossing_objs):
        """
        当相交状态的objs分离了开来，需要将这些分离开的objs与相交前的那些objs进行对应。
        将新一帧与老一帧中处于crossing状态的各个obj相对应，进行更新
        :param after_crossing_objs:
        :param before_crossing_objs:
        :return:
        """
        # 每个位置的before_crossing_obj对应的after_crossing_obj
        for i, old_idx in enumerate(before_crossing_objs):
            old_obj = pick_obj_by_id(self.tracker_record[-1], old_idx)
            distance = []
            for j, new_idx in enumerate(after_crossing_objs):
                new_obj = pick_obj_by_id(self.tracker, new_idx)
                if not new_obj["new_one"]:
                    continue
                distance.append(distance_between_boxes(old_obj["pred_position"], new_obj["position"]))
            # 更新new obj信息
            obj = pick_obj_by_id(self.tracker, after_crossing_objs[np.argmin(distance)])
            obj["id"] = old_obj["id"]
            obj["contains"] = [obj["id"]]
            obj["new_one"] = False
            score = old_obj["score"].copy()
            score.append(obj["score"][0].copy())
            obj["score"] = score
            if len(obj["score"]) > 6:
                obj["score"].pop(0)
            obj["track_score"] = sum(obj["score"])
            obj["class"] = (obj["track_score"] > 0) * 2 - 1
            obj["track_score"] = abs(obj["track_score"])
            obj["show"] = True
            after_crossing_objs[np.argmin(distance)] = obj["id"]
            # obj["pred_cross"] = self.cross_zone
            old_obj["pred_cross"] = 0
        # self.cross_zone += 1

    def normal_update(self, new_idx, match_idx):
        obj = self.tracker[new_idx]
        old_obj = self.tracker_record[-1][match_idx]
        obj["id"] = old_obj["id"]
        score = old_obj["score"].copy()
        score.append(obj["score"][0])
        obj["score"] = score
        if len(obj["score"]) > 6:
            obj["score"].pop(0)
        obj["track_score"] = sum(obj["score"])
        obj["class"] = (obj["track_score"] > 0) * 2 - 1
        obj["track_score"] = abs(obj["track_score"])
        if sum(np.abs(obj["score"])) >= 0.85 * 3:
            obj["show"] = True
        obj["pred_cross"] = 0
        obj["n_occurrence"] = old_obj["n_occurrence"] + 1
        obj["n_absence"] = 0
        obj["pred_position"] = [2 * obj["position"][i] - old_obj["position"][i] for i in range(4)]
        obj["new_one"] = False
        obj["contains"] = [obj["id"]]

    def update_pred_cross(self, pred_cross_info, iou_mat):
        pass

    def update_obj(self, match_info):
        pred_cross_info = []
        for new_idx, match in enumerate(match_info):
            if match["update_type"] == "normal":
                self.normal_update(new_idx, match["match_idx"])
            if match["update_type"] == "pred_cross":
                pred_cross_info.append([match["pred_cross_zone"], new_idx, [match["match_idx"]]])
            if match["update_type"] == "new":
                continue

    def initialize_a_tracker(self, dets, frame):
        self.tracker = []
        if frame == 1:
            idxs_hathead = np.where(dets[0][:, -1] >= self.init_thresh)[0]
            idxs_nohathead = np.where(dets[1][:, -1] >= self.init_thresh)[0]
        else:
            idxs_hathead = np.where(dets[0][:, -1] >= self.update_thresh)[0]
            idxs_nohathead = np.where(dets[1][:, -1] >= self.update_thresh)[0]

        for i, idxs in enumerate([idxs_hathead, idxs_nohathead]):
            for idx in idxs:
                obj = {key: None for key in self.keys}
                obj["unique_id"] = self.unique_id
                self.unique_id += 1
                obj["id"] = self.latest_id
                obj["frame"] = frame
                obj["position"] = [dets[i][idx][0], dets[i][idx][1], dets[i][idx][2], dets[i][idx][3]]
                obj["class"] = -(i * 2 - 1)  # hathead = 1, nohathead = -1
                obj["score"] = [dets[i][idx][-1] * obj["class"]]
                obj["pred_cross"] = 0
                obj["n_absence"] = 0
                obj["pred_position"] = obj["position"]
                obj["track_score"] = 0
                obj["new_one"] = True
                obj["n_occurrence"] = 1
                obj["show"] = False
                obj["crossing"] = False
                obj["contains"] = obj["contains"] = [obj["id"]]
                obj["from_uid"] = 0
                obj["from_iou"] = 0
                self.tracker.append(obj)
                self.latest_id += 1
        self.drop_redundant_obj_by_iou()

    def update_tracker(self, dets, frame):
        self.tracker_record.append(self.tracker)
        self.initialize_a_tracker(dets, frame)

        n_old_obj = len(self.tracker_record[-1])
        n_new_obj = len(self.tracker)

        match_info = [None] * n_new_obj
        # match_info = [{update_type: " ", match_idx: []}, {...}, ... ]
        normal_match = []
        # normal_match = [[new_idx, [old_idx]], ...]
        pred_cross_match = []
        # pred_cross_match = [

        # 构建old_new_iou_mat

        # old_new_iou_mat = np.ndarray((n_new_obj, n_old_obj))
        # for i in range(n_new_obj):
        #     for j in range(n_old_obj):
        #         if self.tracker_record[-1][j]["pred_cross"] == -1:
        #             continue
        #         iou_ij = iou(self.tracker[i]["position"], self.tracker_record[-1][j]["position"])
        #         old_new_iou_mat[i][j] = iou_ij
        #
        # for i in range(n_new_obj):
        #     match_info[i] = []
        #
        #     # iou 从大到小的位置排序
        #     iou_l2s_idx = np.argsort(old_new_iou_mat[i])[::-1]
        #
        #     # 如果没有对应的old obj，则认为这个新obj是新产生的
        #     if old_new_iou_mat[i][iou_l2s_idx[0]] < TRACKING_IOU_THRESH:
        #         match_info[i].append({"update_type": "new", "match_idx": []})
        #
        #     else:
        #         # 如果是普通的old obj，记录下id，之后更新
        #         if old_new_iou_mat[i][iou_l2s_idx[0]] > TRACKING_IOU_THRESH \
        #                 and self.tracker_record[-1][iou_l2s_idx[0]]["pred_cross"] == 0:
        #             match_info[i].append({"update_type": "normal", "match_idx": [iou_l2s_idx[0]]})
        #
        #         # 如果old obj的pred cross为正数，则继续搜寻直到iou < TRACKING_THRESH_IOU。
        #         # 记录下这个new obj和它对应的所有pred cross大于0的old obj，继续循环
        #         pred_cross_zone = self.tracker_record[-1][iou_l2s_idx[0]]["pred_cross"]
        #         if old_new_iou_mat[i][iou_l2s_idx[0]] > TRACKING_IOU_THRESH \
        #                 and pred_cross_zone > 0:
        #             k = 1
        #             match_info[i].append({"update_type": "pred_cross", "match_idx": [iou_l2s_idx[0]],
        #                                   "pred_cross_zone": pred_cross_zone})
        #
        #             while self.tracker_record[-1][iou_l2s_idx[k]]["pred_cross"] == pred_cross_zone \
        #                     and old_new_iou_mat[i][iou_l2s_idx[k]] > TRACKING_IOU_THRESH:
        #                 match_info[i]["match_idx"].append(iou_l2s_idx[k])
        #                 k += 1
        #
        #             if len(match_info[i]["match_idx"]) == 1:
        #                 match_info[i]["update_type"] = "normal"
        #
        # self.update_obj(match_info)

        unmatched_record = [o["id"] for o in self.tracker_record[-1][1:]]
        # crossing_zone = []  # Collection of all the IDs categorized by crossing areas in this frame

        '''
        分析前后帧中各个obj的关系
        '''
        crossing_zone = []
        new_to_crossing_old = [None] * (len(self.tracker_record[-1]) - 1)

        iou_compare = np.zeros((len(self.tracker) - 1, 2)) - 1
        # 这个变量记录了new tracker中各个obj对应的old obj id和相应的iou，他的结构是
        # [[id1, iou1], [id2, iou2], ...]

        drop_redundant_new = []
        # 这个变量记录了在完成最外层循环后，应当删除的new obj

        for idx_new_obj, obj in enumerate(self.tracker[1:]):
            '''
            对于每个new obj，在上一帧搜索与之相近的old obj
            1. 如果old_obj是多个正在相交的物体，先记录new obj的id，在遍历完所有new obj后，根据old obj
            对应的new obj的个数，来判断相交状态是否已经结束。
            2. 如果old obj在上一帧中被预测会进入相交状态，则记录下old obj的id；若在一个new obj下有多个
            old obj与之对应，则认为已经进入了相交状态
            3. 如果其他情况，则直接根据old obj更新new obj
            4. 
            '''
            crossing_id = []
            max_iou = 0
            match_in_old = None
            bbox = obj["position"]
            # Collection of IDs turn from pred_cross to
            for i_tracker_record, old_obj in enumerate(self.tracker_record[-1][1:]):

                if old_obj["pred_cross"] == -1:
                    # 如果这个old_obj处于相交状态，则跳过
                    continue

                old_and_new_iou = iou(bbox, old_obj["position"])
                if old_and_new_iou >= TRACKING_IOU_THRESH:

                    '''
                    如果old_obj是多个正在相交的物体，则先记录下这个new obj的id。
                    注意设置show，crossing和new_one几个参数
                    '''
                    if old_obj["crossing"]:
                        if new_to_crossing_old[i_tracker_record] is None:
                            new_to_crossing_old[i_tracker_record] = [obj['id']]
                        else:
                            new_to_crossing_old[i_tracker_record].append(obj['id'])

                    elif old_obj["pred_cross"] == 0:

                        if old_and_new_iou > max_iou:  # 在这个new obj下，取最大的iou
                            max_iou = old_and_new_iou
                            match_in_old = i_tracker_record
                            iou_compare[idx_new_obj] = [i_tracker_record, old_and_new_iou]

                    else:
                        # old_obj["pred_cross"] > 1， 因为等于-1的情况会在最开始被跳过
                        # 如果上一帧中的某一obj的pred_cross大于0（在上一帧被预测会与其他obj相交），则记录这个obj的id
                        crossing_id.append(old_obj["id"])
                        match_in_old_temp = i_tracker_record

            if len(crossing_id) > 1:
                # 如果有多余1个的old_obj被记录在同一个new obj下，则认为这些old obj正在相交
                obj["crossing"] = True
                obj["show"] = True
                obj["contains"] = [idx for idx in crossing_id]
                crossing_zone.append(crossing_id)
            elif len(crossing_id) == 1:
                #
                match_in_old = match_in_old_temp
                crossing_id = []

            '''
            对于找到了前一帧对应的pred cross为0的obj，更新其信息，但要注意多个new obj对应1个old obj的情况
            '''
            if match_in_old is not None:
                update = True
                # 查找是否存在多对一的情况：
                for new_idx, compare in enumerate(iou_compare):
                    if new_idx != idx_new_obj and match_in_old == compare[0]:
                        if max_iou < compare[1]:
                            drop_redundant_new.append(idx_new_obj)
                            update = False
                            break
                        else:
                            drop_redundant_new.append(new_idx)
                            break
                if update:
                    unmatched_record[match_in_old] = 0
                    old_obj = self.tracker_record[-1][match_in_old + 1]
                    obj["id"] = old_obj["id"]
                    score = old_obj["score"].copy()
                    score.append(obj["score"][0])
                    obj["score"] = score
                    if len(obj["score"]) > 6:
                        obj["score"].pop(0)
                    obj["track_score"] = sum(obj["score"])
                    obj["class"] = (obj["track_score"] > 0) * 2 - 1
                    obj["track_score"] = abs(obj["track_score"])
                    if sum(np.abs(obj["score"])) >= 0.85 * 3:
                        obj["show"] = True
                    obj["frame"] = frame
                    obj["position"] = bbox
                    obj["pred_cross"] = 0
                    obj["n_occurrence"] = old_obj["n_occurrence"] + 1
                    obj["n_absence"] = 0
                    obj["pred_position"] = [2 * obj["position"][i] - old_obj["position"][i] for i in range(4)]
                    obj["new_one"] = False
                    obj["contains"] = [obj["id"]]

        '''
        对于那些与处于crossing状态的obj相近的new obj，分两类处理：
        1. 如果一个old obj下之对应一个new obj，则认为这个new obj是old obj在新一帧中的继承
        2. 如果一个old obj下对应多余一个new obj，则认为这个old obj离开了crossing状态，并产生了这些new obj
        2.1 将这些new obj根据就近原则与进入crossing状态之前的那些obj进行匹配
        '''
        for old_idx, crossing_new in enumerate(new_to_crossing_old):
            if crossing_new is not None and len(crossing_new) == 1:
                for obj in self.tracker[1:]:
                    if obj['id'] == crossing_new[0]:
                        old_obj = self.tracker_record[-1][old_idx + 1]
                        obj["id"] = old_obj["id"]
                        score = old_obj["score"].copy()
                        score.append(obj["score"][0])
                        obj["score"] = score
                        if len(obj["score"]) > 6:
                            obj["score"].pop(0)
                        obj["track_score"] = sum(obj["score"])
                        obj["class"] = (obj["track_score"] > 0) * 2 - 1
                        obj["track_score"] = abs(obj["track_score"])
                        obj["show"] = True
                        obj["crossing"] = True
                        obj["frame"] = frame
                        obj["pred_cross"] = 0
                        obj["n_occurrence"] = old_obj["n_occurrence"] + 1
                        obj["n_absence"] = 0
                        obj["pred_position"] = [2 * obj["position"][i] - old_obj["position"][i] for i in range(4)]
                        obj["new_one"] = False
                        obj["contains"] = old_obj["contains"]
            elif crossing_new is not None and len(crossing_new) > 1:
                contains = self.tracker_record[-1][old_idx + 1]['contains']
                self.match_after_crossing(crossing_new, contains)

        '''
        当有多个new objs对应一个pred cross = 0 的old obj时，删除多余的new objs
        删除pred_cross为-1，且其对应的crossing框已经消失的obj
        '''
        new_tracker = self.tracker.copy()
        self.tracker = []
        for idx_new, new_obj in enumerate(new_tracker):
            if idx_new not in drop_redundant_new:
                self.tracker.append(new_obj)

        '''
        将已进入或者新进入crossing状态的object稍作修改，放入新的tracker中，但不让其显示。
        （即将pred_cross=-1且其对应的crossing框未消失的obj放入新的tracker）
        '''
        for i_tracker_record, old_obj in enumerate(self.tracker_record[-1][1:]):
            if sum([old_obj["id"] in zone for zone in crossing_zone]) or old_obj["pred_cross"] == -1:
                unmatched_record[i_tracker_record] = 0
                old_obj_copy = old_obj.copy()
                old_obj_copy["n_absence"] += 1
                old_obj_copy["position"] = old_obj_copy["pred_position"]
                old_obj_copy["pred_position"] = [2 * old_obj_copy["position"][i] - old_obj["position"][i] for i in
                                                 range(4)]
                # old_obj_copy["score"].append(0)
                # if len(old_obj_copy["score"]) > 6:
                #     old_obj_copy["score"].pop(0)
                old_obj_copy["track_score"] = sum(old_obj_copy["score"])
                old_obj_copy["pred_cross"] = -1
                old_obj_copy["show"] = False
                self.tracker.append(old_obj_copy)

        '''
        对于没有在最新一帧中找到对应object的old object，认为其短暂消失了，只要其没有连续消失n帧，则将这些old object放入新的tracker中。
        '''
        for old_obj in self.tracker_record[-1][1:]:

            if old_obj["id"] in unmatched_record and old_obj["n_absence"] < MAX_ABSENT_FRAME \
                    and old_obj["show"] and not old_obj["crossing"]:

                old_obj_copy = old_obj.copy()
                old_obj_copy["n_absence"] += 1
                old_obj_copy["score"].append(0)
                old_obj_copy["position"] = old_obj_copy["pred_position"]
                old_obj_copy["pred_position"] = [2 * old_obj_copy["position"][i] - old_obj["position"][i] for i in
                                                 range(4)]
                if len(old_obj_copy["score"]) > 6:
                    old_obj_copy["score"].pop(0)
                old_obj_copy["track_score"] = sum(old_obj_copy["score"])
                self.tracker.append(old_obj_copy)

        '''
        检查是否有可能相交的obj
        '''
        cross_id = []
        for i, obj1 in enumerate(self.tracker[:-1]):
            if obj1["pred_cross"] or obj1["crossing"]:
                continue
            for obj2 in self.tracker[i + 1:]:
                if obj2["pred_cross"]:
                    continue
                pred_iou = iou(obj1["pred_position"], obj2["pred_position"])
                if pred_iou >= CROSSING_IOU_THRESH:
                    cross_id.append((obj1["id"], obj2["id"]))
                    obj1["pred_cross"] = obj2["pred_cross"] = self.cross_zone
            if len(cross_id):
                self.cross_zone += 1

        self.latest_id = self.find_latest_id()

    def show(self):
        # Use for debug
        for idx, obj in enumerate(self.tracker):
            obj_for_show = obj.copy()
            print("C:", "%-3s" % str(obj_for_show["contains"]),
                  "%-6s" % labels_to_names[obj_for_show["class"]],
                  "Score:", "%-44s" % str(np.round(obj_for_show["score"], 3)),
                  "Actual:", "%-20s" % str(np.round(obj_for_show["position"], 0)),
                  "Pred:", "%-20s" % str(np.round(obj_for_show["pred_position"], 0)),
                  "Show:", "%-5s" % obj_for_show["show"],
                  "pred_cross", "%-2s" % obj_for_show["pred_cross"],
                  "Crossing", "%-5s" % obj_for_show["crossing"],
                  "Occ:", "%-2s" % str(obj_for_show["n_occurrence"]),
                  "Absence:", "%-2s" % str(obj_for_show["n_absence"]),
                  "Index:", "%-2s" % str(idx))


def show_for_debug(tracker):
    for idx, obj in enumerate(tracker):
        obj_for_show = obj.copy()
        print("C:", "%-3s" % str(obj_for_show["contains"]),
              "%-6s" % labels_to_names[obj_for_show["class"]],
              "Score:", "%-44s" % str(np.round(obj_for_show["score"], 3)),
              "Actual:", "%-20s" % str(np.round(obj_for_show["position"], 0)),
              "Pred:", "%-20s" % str(np.round(obj_for_show["pred_position"], 0)),
              "Show:", "%-5s" % obj_for_show["show"],
              "pred_cross", "%-2s" % obj_for_show["pred_cross"],
              "Crossing", "%-5s" % obj_for_show["crossing"],
              "Occ:", "%-2s" % str(obj_for_show["n_occurrence"]),
              "Absence:", "%-2s" % str(obj_for_show["n_absence"]),
              "Index:", "%-2s" % str(idx))
