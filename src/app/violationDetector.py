class ViolationDetector:
    def __init__(self, slots, classes) -> None:
        self.slots = slots
        self.names = classes
        self.allowed_vehicles = ['car', 'truck', 'bus', 'person']

    def detect(self, vehicles):
        overlaps = []
        foreigns = []
        for vehicle in vehicles:
            occupied_slots = []
            for slot_id in self.slots:
                #if self.overlap(*slot_id[1:], *vehicle[:4]):
                if self.overlap(*self.slots[slot_id], *vehicle[:4]):
                    occupied_slots.append(slot_id)
            if len(occupied_slots) > 1:
                overlaps += occupied_slots
            
            # # foreign objects detection
            vehicle_type = self.names[int(vehicle[5])]
            if len(occupied_slots) > 0:
                if vehicle_type not in self.allowed_vehicles:
                    foreign = [vehicle_type, occupied_slots]
                    foreigns.append(foreign)
        
        return overlaps, foreigns

    def overlap(self, x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2):
        xA = max(x1_1, x1_2)
        yA = max(y1_1, y1_2)
        xB = min(x2_1, x2_2)
        yB = min(y2_1, y2_2)
        intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        boxBArea = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
        ioa = intersectionArea / min(boxAArea, boxBArea)
        return ioa > 0.4
