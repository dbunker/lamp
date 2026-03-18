d9(o0).
d0(o5).
d5(o1).
d0(o1).
d3(o4).
d7(o3).

d6(V1) :- d5(V1).
d2(V1) :- d5(V1).
d8(V1) :- d7(V1); not d0(V1).
d7(V1) :- d0(V1); not d5(V1).