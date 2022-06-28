import simpy
import random
import matplotlib.pyplot as plt


from framework.utils import Location
from framework.Node import Node, EnergyProfile
from framework.Gateway import Gateway
from framework.TransmissionInterface import AirInterface
from framework.Backend import Server
from framework.LoRaParameters import LoRaParameters
from framework.Environment import *
from config import *


class LoRaCommunication:
    def __init__(self, node_indexes, gateway_indexes, step_time, environment, distance_scale, offset=2000):
        self.nodes = []
        self.gateways = []
        assert step_time >= offset + 3000
        self.step_time = step_time
        self.offset = offset
        self.sim_env = simpy.Environment()
        self.channel_nodes = {}
        for channel in range(Gateway.NO_CHANNELS):
            self.channel_nodes[channel] = []
        self.environment=environment
        self.steps = 0

        self.server = Server(self.gateways, self.sim_env)
        self.air_interface = AirInterface(self.sim_env, self.gateways, self.server)


        lora_para = [LoRaParameters(i % Gateway.NO_CHANNELS, sf=random.choice(LoRaParameters.SPREADING_FACTORS)) for i in range(len(node_indexes))]
        for i, idx in enumerate(node_indexes):
            node = Node(i, EnergyProfile(0.1), lora_para[i],
                                   self.air_interface, self.sim_env, Location(idx[0]*environment.grass_r.nsres * distance_scale, idx[1]*environment.grass_r.ewres * distance_scale), idx, True)
            self.channel_nodes[lora_para[i].channel].append(node)
            self.nodes.append(node)
        for i, idx in enumerate(gateway_indexes):
            self.gateways.append(Gateway(i, Location(idx[0]*environment.grass_r.nsres* distance_scale, idx[1]*environment.grass_r.ewres *distance_scale), idx, self.sim_env))


    def step(self, actions, skip_lora=False):
        assert len(self.nodes) == len(actions)
        assert self.sim_env.now == self.steps * self.step_time
        self.steps += 1
        send_index = [idx for idx, send in enumerate(actions) if send]
        for i in range(len(self.nodes)):
            self.sim_env.process(self._node_send_sensed_value(i, actions[i], skip_lora))
        self.sim_env.run(self.step_time * self.steps)
        received = []
        for i in send_index:
            if self.nodes[i].packet_to_send.received:
                received.append(i)
        return send_index, received

    def _node_send_sensed_value(self, node_index, send, skip_lora):
        node = self.nodes[node_index]
        data = node.sense(self.environment)
        yield self.sim_env.timeout(np.random.randint(self.offset))
        if send:
            data['x'] = node.location.x
            data['y'] = node.location.y
            packet = node.create_unique_packet(data, False, False)
            yield self.sim_env.process(node.send(packet, skip_lora))
        yield self.sim_env.timeout(self.step_time * self.steps - self.sim_env.now)

    def _node_send_test(self, node_index):
        yield self.sim_env.timeout(np.random.randint(self.offset))
        node = self.nodes[node_index]
        packet = node.create_unique_packet(None, True, True)
        yield self.sim_env.process(node.send(packet))
        yield self.sim_env.timeout(self.step_time * self.steps - self.sim_env.now)

    def get_grass_time(self):
        return self.steps * self.step_time * SIMPY_TO_GRASS_TIME_FACTOR

    def pre_adr(self, rounds: int, show=False, percentage=0.8):
        assert rounds > 50
        N = int(len(self.nodes) * percentage)
        for i in range(len(self.nodes)):
            self.nodes[i].adr = True
        record_per = []
        for i in range(rounds):
            assert self.sim_env.now == self.steps * self.step_time
            self.steps += 1
            send = []
            for channel in self.channel_nodes:
                list_of_nodes = self.channel_nodes[channel]
                send.extend(random.sample(list_of_nodes, int(percentage*len(list_of_nodes))))
            send_node = [n.id for n in send]
            for j in send_node:
                self.sim_env.process(self._node_send_test(j))
            self.sim_env.run(self.step_time * self.steps)
            count = 0
            for j, idx in enumerate(send_node):
                if self.nodes[idx].last_packet_success:
                    count += 1
            record_per.append(1 - count/len(send_node))
        latest_per = record_per[-50:]

        count = 0
        threshold = 0.3
        while not self._check_adr_convergence(latest_per, threshold):
            for i in range(50):
                assert self.sim_env.now == self.steps * self.step_time
                self.steps += 1
                send_node = random.sample(range(len(self.nodes)), N)
                for j in send_node:
                    self.sim_env.process(self._node_send_test(j))
                self.sim_env.run(self.step_time * self.steps)
                count = 0
                for j, idx in enumerate(send_node):
                    if self.nodes[idx].last_packet_success:
                        count +=1
                latest_per[i] = 1 - count/len(send_node)
            count += 1
            if count == int(10/percentage) or count == 30:
                self.reset(True)
                plt.figure(figsize=(5, 5))
                plt.scatter(range(len(record_per)), record_per)
                plt.title("Running ADR...")
                plt.xlabel("iterations")
                plt.ylabel("Packet error rate")
                plt.show()
                return self.pre_adr(rounds, show, percentage)
            record_per.extend(latest_per)
        if np.average(record_per[-50:-1]) > 0.6:
            self.reset(True)
            plt.figure(figsize=(5, 5))
            plt.scatter(range(len(record_per)), record_per)
            plt.title("Running ADR...")
            plt.xlabel("iterations")
            plt.ylabel("Packet error rate")
            plt.show()
            return self.pre_adr(rounds, show, percentage)
        if show:
            plt.figure(figsize=(5,5))
            plt.scatter(range(len(record_per)), record_per)
            plt.title("Running ADR...")
            plt.xlabel("iterations")
            plt.ylabel("Packet error rate")
        for i in range(len(self.nodes)):
            self.nodes[i].adr = False
        return record_per

    def reset(self, reset_lora=False):
        self.sim_env = simpy.Environment()
        self.steps = 0
        for i, node in enumerate(self.nodes):
            node.reset(self.sim_env)
            if reset_lora:
                self.nodes[i].para = LoRaParameters(i % Gateway.NO_CHANNELS, sf=random.choice(LoRaParameters.SPREADING_FACTORS))
        for i in range(len(self.gateways)):
            self.gateways[i].reset(self.sim_env)
        self.server.reset(self.sim_env)
        self.air_interface.reset(self.sim_env)

    def _check_adr_convergence(self, mean, difference):
        return np.max(mean) - np.min(mean) < difference