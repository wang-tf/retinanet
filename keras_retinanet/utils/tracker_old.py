import numpy as np

REDUNDANT_IOU_THRESH = 0.8
CROSSING_IOU_THRESH = 0.0
TRACKING_IOU_THRESH = 0.0
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


def predict_position(current_position, previous_position):
    xp0 = previous_position[0]
    xp1 = previous_position[2]
    yp0 = previous_position[1]
    yp1 = previous_position[3]
    xc0 = current_position[0]
    xc1 = current_position[2]
    yc0 = current_position[1]
    yc1 = current_position[3]

    previous_center = [(xp0 + xp1) / 2, (yp0 + yp1) / 2]
    previous_size = [xp1 - xp0, yp1 - yp0]
    current_center = [(xc0 + xc1) / 2, (yc0 + yc1) / 2]
    current_size = [xc1 - xc0, yc1 - yc0]

    pred_center = [2 * current_center[0] - previous_center[0],
                   2 * current_center[1] - previous_center[1]]
    pred_size = [(previous_size[0] + current_size[0]) / 2,
                 (previous_size[1] + current_size[1]) / 2]

    pred_position = [max(pred_center[0] - pred_size[0] / 2, 0),
                     max(pred_center[1] - pred_size[1] / 2, 0),
                     pred_center[0] + pred_size[0] / 2,
                     pred_center[1] + pred_size[1] / 2]
    return pred_position


class Tracker(object):
    def __init__(self):
        self.keys = ["unique_id", "id", "frame", "position", "class", "score", "pred_cross", "n_absence",
                     "pred_position", "track_score", "new_one", "n_occurrence", "show", "crossing",
                     "contains", "from_uid", "from_iou"]
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
        for obj in self.tracker:
            if obj["id"] > latest_id:
                latest_id = obj["id"]
        latest_id += 1
        return latest_id

    def get_recorded_obj_by_id(self, index, frame):
        for obj in self.tracker_record[frame - 1]:
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

    def match_after_crossing(self, after_crossing_objs, before_crossing_objs, drop_new, _separated_crossing, _old_idx):
        """
        当相交状态的objs分离了开来，需要将这些分离开的objs与相交前的那些objs进行对应。
        将新一帧与老一帧中处于crossing状态的各个obj相对应，进行更新
        :param after_crossing_objs:
        :param before_crossing_objs:
        :return:
        """
        # 每个位置的before_crossing_obj对应的after_crossing_obj
        distance_mat = np.empty((len(after_crossing_objs), len(before_crossing_objs)))
        for i, old_idx in enumerate(before_crossing_objs):
            old_obj = pick_obj_by_id(self.tracker_record[-1], old_idx)
            for j, new_idx in enumerate(after_crossing_objs):
                new_obj = pick_obj_by_id(self.tracker, new_idx)
                distance_mat[j][i] = distance_between_boxes(old_obj["pred_position"], new_obj["position"])

        # 更新new obj信息
        pairs = min(len(after_crossing_objs), len(before_crossing_objs))
        for pair in range(pairs):
            min_idx = np.argmin(distance_mat)
            r = int(min_idx / distance_mat.shape[1])
            c = min_idx % distance_mat.shape[1]
            obj = pick_obj_by_id(self.tracker, after_crossing_objs[r])
            old_obj = pick_obj_by_id(self.tracker_record[-1], before_crossing_objs[c])
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
            # obj["pred_cross"] = self.cross_zone
            old_obj["pred_cross"] = 0

            # update distance mat
            distance_mat = np.delete(distance_mat, r, 0)
            distance_mat = np.delete(distance_mat, c, 1)
            after_crossing_objs = after_crossing_objs[:r] + after_crossing_objs[r + 1:]
            before_crossing_objs = before_crossing_objs[:c] + before_crossing_objs[c + 1:]

        if len(after_crossing_objs):
            for idx in after_crossing_objs:
                for j, tracker in enumerate(self.tracker):
                    if tracker['id'] == idx:
                        drop_new.append(j)

        if len(before_crossing_objs) == 0:
            _separated_crossing.append(_old_idx)
        return drop_new, _separated_crossing
        # self.cross_zone += 1

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

    def find_unmatched_record(self):
        pass

    def update_tracker(self, dets, frame):
        self.tracker_record.append(self.tracker)
        self.initialize_a_tracker(dets, frame)

        '''
        建立new_old_iou_mat
        '''
        n_old_obj = len(self.tracker_record[-1])
        n_new_obj = len(self.tracker)
        old_new_iou_mat = np.zeros((n_new_obj, n_old_obj))
        for i in range(n_new_obj):
            for j in range(n_old_obj):
                if self.tracker_record[-1][j]["pred_cross"] == -1:
                    continue
                iou_ij = iou(self.tracker[i]["position"], self.tracker_record[-1][j]["position"])
                old_new_iou_mat[i][j] = iou_ij

        '''
        分析前后帧中各个obj的关系
        '''
        separated_crossing = []
        new_to_crossing_old = [None] * (len(self.tracker_record[-1]))
        unmatched_record = [o["id"] for o in self.tracker_record[-1]]

        # 这个变量记录了new tracker中各个obj对应的old obj id和相应的iou，他的结构是
        # [[id1, iou1], [id2, iou2], ...]
        iou_compare = np.zeros((len(self.tracker), 2)) - 1

        # 这个变量记录了在完成最外层循环后，应当删除的new obj
        drop_redundant_new = []

        for idx_new_obj, obj in enumerate(self.tracker):
            '''  
            对于每个new obj，在上一帧搜索与之相近的old obj
            1. 如果old_obj是多个正在相交的物体，先记录new obj的id，在遍历完所有new obj后，根据old obj
            对应的new obj的个数，来判断相交状态是否已经结束。
            2. 如果old obj的pred_cross > 0，则记录下old obj的id；若在一个new obj下有多个
            old obj与之对应，则认为已经进入了相交状态
            3. 如果其他情况，则直接根据old obj更新new obj
            4. 
            '''
            max_iou = 0
            match_in_old = None
            bbox = obj["position"]

            if sum(old_new_iou_mat[idx_new_obj]) == 0:
                continue

            iou_l2s_idx = np.argsort(old_new_iou_mat[idx_new_obj])[::-1]
            if self.tracker_record[-1][iou_l2s_idx[0]]["crossing"] and \
                    self.tracker_record[-1][iou_l2s_idx[0]]["pred_cross"] == 0:
                k = 1
                if new_to_crossing_old[iou_l2s_idx[0]] is None:
                    new_to_crossing_old[iou_l2s_idx[0]] = [obj['id']]
                else:
                    new_to_crossing_old[iou_l2s_idx[0]].append(obj['id'])
                while iou_l2s_idx[k] > TRACKING_IOU_THRESH and self.tracker_record[-1][iou_l2s_idx[k]]["crossing"]:
                    new_to_crossing_old[iou_l2s_idx[0]].append(obj['id'])

            elif self.tracker_record[-1][iou_l2s_idx[0]]["pred_cross"] >= 0:
                max_iou = old_new_iou_mat[idx_new_obj][iou_l2s_idx[0]]
                match_in_old = iou_l2s_idx[0]
                iou_compare[idx_new_obj] = [match_in_old, max_iou]

            # for idx_old_obj, old_obj in enumerate(self.tracker_record[-1]):
            #
            #     if old_obj["pred_cross"] == -1:
            #         # 如果这个old_obj处于相交状态，则跳过
            #         continue
            #
            #     # 取这个new obj对应的所有old obj中iou最大的
            #
            #     # 如果这个old obj是多个正在相交的物体，则先记录下这个new obj的id
            #     old_and_new_iou = old_new_iou_mat[idx_new_obj][idx_old_obj]
            #     if old_and_new_iou > TRACKING_IOU_THRESH:
            #         '''
            #         如果old_obj是多个正在相交的物体，则先记录下这个new obj的id。
            #         注意设置show，crossing和new_one几个参数
            #         '''
            #         if old_obj["crossing"]:
            #             if new_to_crossing_old[idx_old_obj] is None:
            #                 new_to_crossing_old[idx_old_obj] = [obj['id']]
            #             else:
            #                 new_to_crossing_old[idx_old_obj].append(obj['id'])
            #
            #         elif old_obj["pred_cross"] >= 0:
            #
            #             if old_and_new_iou > max_iou:  # 在这个new obj下，取最大的iou
            #                 max_iou = old_and_new_iou
            #                 match_in_old = idx_old_obj
            #                 iou_compare[idx_new_obj] = [idx_old_obj, old_and_new_iou]

            '''
            对于找到了前一帧对应的pred cross为0的obj，更新其信息，但要注意多个new obj对应1个old obj的情况
            '''
            if match_in_old is not None:
                update = True
                # 查找是否存在多对一的情况，若存在，则之后会舍弃多余的new obj
                for new_idx, compare in enumerate(iou_compare):
                    if idx_new_obj == 0:
                        break
                    if new_idx != idx_new_obj and match_in_old == compare[0]:
                        if max_iou < compare[1]:
                            drop_redundant_new.append(idx_new_obj)
                            iou_compare[idx_new_obj] = [-1, -1]
                            update = False
                            break
                        else:
                            drop_redundant_new.append(new_idx)
                            iou_compare[new_idx] = [-1, -1]
                            break
                # 若是普通的一对一，则直接更新
                if update:
                    unmatched_record[match_in_old] = -1
                    old_obj = self.tracker_record[-1][match_in_old]
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
                    obj["pred_position"] = predict_position(obj['position'], old_obj['position'])
                    obj["new_one"] = False
                    obj["contains"] = old_obj["contains"].copy()

        '''
        对于那些与处于crossing状态的obj相近的new obj，分两类处理：
        1. 如果一个old obj下之对应一个new obj，则认为这个new obj是old obj在新一帧中的继承
        2. 如果一个old obj下对应多余一个new obj，则认为这个old obj离开了crossing状态，并产生了这些new obj
        2.1 将这些new obj根据就近原则与进入crossing状态之前的那些obj进行匹配
        '''

        for old_idx, crossing_new in enumerate(new_to_crossing_old):
            if crossing_new is None:
                continue
            elif len(crossing_new) == 1:
                for obj in self.tracker:
                    if obj['id'] == crossing_new[0]:
                        unmatched_record[old_idx] = -1
                        old_obj = self.tracker_record[-1][old_idx]
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
                        obj["pred_cross"] = old_obj["pred_cross"]
                        obj["n_occurrence"] = old_obj["n_occurrence"] + 1
                        obj["n_absence"] = 0
                        obj["pred_position"] = predict_position(obj['position'], old_obj['position'])
                        obj["new_one"] = False
                        obj["contains"] = old_obj["contains"].copy()
            elif len(crossing_new) > 1:
                contains = self.tracker_record[-1][old_idx]['contains']
                drop_redundant_new, separated_crossing = self.match_after_crossing(crossing_new, contains,
                                                                                   drop_redundant_new,
                                                                                   separated_crossing, old_idx)

        '''
        对于没有在最新一帧中找到对应object的old object，
        若pred_cross为0，则认为其短暂消失了，只要其没有连续消失n帧，则将这些old object放入新的tracker中。
        若pred_cross为正数，则将其补入iou最大的那个obj
        '''
        append_id = []
        for i, old_obj in enumerate(self.tracker_record[-1]):
            if old_obj['id'] not in unmatched_record:
                continue
            elif old_obj["n_absence"] < MAX_ABSENT_FRAME and old_obj["show"] and old_obj["pred_cross"] == 0:
                if i not in separated_crossing:
                    old_obj_copy = old_obj.copy()
                    old_obj_copy["n_absence"] += 1
                    old_obj_copy["score"].append(0)
                    old_obj_copy["position"] = old_obj_copy["pred_position"]
                    old_obj_copy["pred_position"] = predict_position(old_obj_copy['position'], old_obj['position'])
                    if len(old_obj_copy["score"]) > 6:
                        old_obj_copy["score"].pop(0)
                    if sum(np.abs(old_obj_copy["score"])) >= 0.85 * 1:
                        old_obj_copy["show"] = True
                    old_obj_copy["track_score"] = sum(old_obj_copy["score"])
                    self.tracker.append(old_obj_copy)

            elif old_obj["pred_cross"] > 0:
                # 更新
                if max(old_new_iou_mat[:, i]) <= 0:
                    old_obj_copy = old_obj.copy()
                    old_obj_copy["pred_cross"] = 0
                    old_obj_copy["n_absence"] += 1
                    old_obj_copy["score"].append(0)
                    old_obj_copy["position"] = old_obj_copy["pred_position"]
                    old_obj_copy["pred_position"] = predict_position(old_obj_copy['position'], old_obj['position'])
                    if len(old_obj_copy["score"]) > 6:
                        old_obj_copy["score"].pop(0)
                    if sum(np.abs(old_obj_copy["score"])) >= 0.85 * 1:
                        old_obj_copy["show"] = True
                    old_obj_copy["track_score"] = sum(old_obj_copy["score"])
                    self.tracker.append(old_obj_copy)
                else:
                    new_idx = np.argmax(old_new_iou_mat[:, i])
                    new_obj = self.tracker[new_idx]
                    if new_idx in drop_redundant_new:
                        del (drop_redundant_new[drop_redundant_new.index(new_idx)])
                        new_obj["contains"] = [old_obj['id']]
                        new_obj['id'] = old_obj['id']
                        score = old_obj["score"].copy()
                        score.append(new_obj["score"][0])
                        new_obj["score"] = score
                        if len(new_obj["score"]) > 6:
                            new_obj["score"].pop(0)
                        new_obj["track_score"] = sum(new_obj["score"])
                        new_obj["class"] = (new_obj["track_score"] > 0) * 2 - 1
                        new_obj["track_score"] = abs(new_obj["track_score"])
                        if sum(np.abs(new_obj["score"])) >= 0.85 * 1:
                            new_obj["show"] = True
                        new_obj["frame"] = frame
                        new_obj["n_occurrence"] = old_obj["n_occurrence"] + 1
                        new_obj["n_absence"] = 0
                        new_obj["pred_position"] = predict_position(new_obj['position'], old_obj['position'])
                        new_obj["new_one"] = False
                    else:
                        for contain in old_obj['contains']:
                            new_obj["contains"].append(contain)
                        new_obj["crossing"] = True
                        new_obj["n_occurrence"] = 1
                        new_obj["show"] = True
                        new_obj["id"] = self.latest_id
                        self.latest_id += 1
                        new_obj["pred_cross"] = 0
                        new_obj["score"] = [new_obj["score"][-1]]
                        # 将相交的物体补上
                        append_id.append(new_obj["contains"])

        '''
        将已进入或者新进入crossing状态的object稍作修改，放入新的tracker中，但不让其显示。
        '''
        for idx_old_obj, old_obj in enumerate(self.tracker_record[-1]):
            if sum([old_obj["id"] in zone for zone in append_id]) or old_obj["pred_cross"] == -1:
                unmatched_record[idx_old_obj] = -1
                old_obj_copy = old_obj.copy()
                old_obj_copy["n_absence"] += 1
                old_obj_copy["position"] = old_obj_copy["pred_position"]
                old_obj_copy["pred_position"] = predict_position(old_obj_copy['position'], old_obj['position'])
                # old_obj_copy["score"].append(0)
                # if len(old_obj_copy["score"]) > 6:
                #     old_obj_copy["score"].pop(0)
                old_obj_copy["track_score"] = sum(old_obj_copy["score"])
                old_obj_copy["pred_cross"] = -1
                old_obj_copy["show"] = False
                self.tracker.append(old_obj_copy)

        '''
        当有多个new objs对应一个pred cross = 0 的old obj时，删除多余的new objs
        当obj的pred_cross = -1且其对应的crossing框已经消失，则删除这个obj
        '''
        new_tracker = self.tracker.copy()
        self.tracker = []
        crossing_obj = []
        for idx_new, new_obj in enumerate(new_tracker):
            if len(new_obj["contains"]) > 1:
                crossing_obj.append(new_obj["contains"])
            if idx_new not in drop_redundant_new:
                # 如果pred_cross = -1，且无对应的crossing框，则删除这个obj
                if new_obj["pred_cross"] == -1 and not sum(
                        [new_obj["id"] in crossing for crossing in crossing_obj]):
                    continue
                self.tracker.append(new_obj)

        '''
        检查是否有可能相交的obj
        '''
        cross_id = []
        for i, obj1 in enumerate(self.tracker[:-1]):
            if obj1["pred_cross"] or not obj1['show']:
                continue
            for obj2 in self.tracker[i + 1:]:
                if obj2["pred_cross"] or not obj1['show']:
                    continue
                pred_iou = iou(obj1["pred_position"], obj2["pred_position"])
                if pred_iou > CROSSING_IOU_THRESH:
                    cross_id.append((obj1["id"], obj2["id"]))
                    obj1["pred_cross"] = obj2["pred_cross"] = self.cross_zone
            if len(cross_id):
                self.cross_zone += 1

        self.latest_id = self.find_latest_id()
        # self.drop_redundant_tracker_by_id()

    def show(self):
        # Use for debug
        for idx, obj in enumerate(self.tracker):
            obj_for_show = obj.copy()
            print("%-11s" % str(obj_for_show["contains"]),
                  "%-6s" % labels_to_names[obj_for_show["class"]],
                  "Actual:", "%-20s" % str(np.round(obj_for_show["position"], 0)),
                  "Pred:", "%-20s" % str(np.round(obj_for_show["pred_position"], 0)),
                  "Show:", "%-5s" % obj_for_show["show"],
                  "pred_cross", "%-2s" % obj_for_show["pred_cross"],
                  "Crossing", "%-5s" % obj_for_show["crossing"],
                  "Occ:", "%-2s" % str(obj_for_show["n_occurrence"]),
                  "Absence:", "%-2s" % str(obj_for_show["n_absence"]),
                  "Id:", "%-2s" % str(obj_for_show["id"]),
                  "Score:", "%-44s" % str(np.round(obj_for_show["score"], 3)))


def show_for_debug(tracker):
    for idx, obj in enumerate(tracker):
        obj_for_show = obj.copy()
        print("%-11s" % str(obj_for_show["contains"]),
              "%-6s" % labels_to_names[obj_for_show["class"]],
              "Actual:", "%-20s" % str(np.round(obj_for_show["position"], 0)),
              "Pred:", "%-20s" % str(np.round(obj_for_show["pred_position"], 0)),
              "Show:", "%-5s" % obj_for_show["show"],
              "pred_cross", "%-2s" % obj_for_show["pred_cross"],
              "Crossing", "%-5s" % obj_for_show["crossing"],
              "Occ:", "%-2s" % str(obj_for_show["n_occurrence"]),
              "Absence:", "%-2s" % str(obj_for_show["n_absence"]),
              "Id:", "%-2s" % str(obj_for_show["id"]),
              "Score:", "%-44s" % str(np.round(obj_for_show["score"], 3)))
