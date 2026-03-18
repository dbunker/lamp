d2(o2).
d2(o0).
d5(o4).
d9(o1).
d4(o4).
d2(o4).
d8(o0).
d6(o4).
d8(o4).

d2(V1) :- d9(V1).
d3(V1) :- d7(V1).
d7(V1) :- d8(V1); not d5(V1).
d5(V1) :- d2(V1); not d8(V1).