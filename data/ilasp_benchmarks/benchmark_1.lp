d2(o3).
d0(o7).
d9(o8).
d8(o0).
d3(o4).
d9(o7).
d4(o2).
d2(o2).
d7(o7).

d5(V1) :- d2(V1).
d1(V1) :- d3(V1).
d9(V1) :- d5(V1).
d0(V1) :- d1(V1); not d2(V1).
d1(V1) :- d9(V1); not d4(V1); not d7(V1).
d6(V1) :- d9(V1); not d5(V1); not d7(V1).