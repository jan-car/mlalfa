# -*- coding: utf-8 -*-
# 2021 Machine Learning Course Alfatraining
# Author: J. Caron
#
# Implementation from Frochte - Maschinelles Lernen - 6.1
# PURELY Frochte Code, just copied, not used atm...
#

import logging
import numpy as np


class Tree:

    _log = logging.getLogger(__name__ + '.Tree')

    def __init__(self, varNo, value, operator):
        self.rootNode = TreeNode(0, value, varNo=varNo, operator=operator)
        self.nodes = []
        self.nodes.append(self.rootNode)
        self.leafNodes = []
        self.leafNodes.append(0)

    def addNode(self, ChildOf, branch, value, operator='<', varNo=0):
        node = TreeNode(len(self.nodes), value, ChildOf=ChildOf, operator=operator, varNo=varNo)
        self.leafNodes.append(node.number)
        self.nodes.append(node)
        parent = self.nodes[ChildOf]
        if branch is True:
            parent.leftTrue = node
        else:
            parent.rightFalse = node
        if parent.leftTrue is not None and parent.rightFalse is not None:
            toDelete = self.leafNodes.index(parent.number)
            del self.leafNodes[toDelete]
        return node.number

    def trace(self, x):
        traceRoute = self.rootNode.trace(x)[0]
        return traceRoute

    def eval(self, x):
        traceRoute = self.trace(x)
        y = np.zeros(len(traceRoute))
        for i in range(len(y)):
            y[i] = self.nodes[traceRoute[i][-1]]()
        return y

    def evalCondition(self, x):
        if self.operator == '=':
            cond = x[:, self.varNo] == self.value
        elif self.operator == '<':
            cond = x[:, self.varNo] < self.value
        else:  # case >
            cond = x[:, self.varNo] > self.value
        return cond


class TreeNode:

    _log = logging.getLogger(__name__ + '.TreeNode')

    def __init__(self, number, value, ChildOf=None, operator='<', varNo=0):
        self.number = number
        self.childOf = ChildOf
        self.leftTrue = None
        self.rightFalse = None
        self.value = value
        self.varNo = varNo
        self.operator = operator

    def __call__(self):
        """liefert die Zahl oder ein anderes Objekt, das dem Knoten zugeordnet ist"""
        return self.value

    def leafNode(self):
        """ liefert Informationen, ob der Knoten ein Endpunkt (Blatt) ist"""
        if self.leftTrue is not None and self.rightFalse is not None:
            return False
        else:
            return True
