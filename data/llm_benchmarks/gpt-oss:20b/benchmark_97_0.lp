d4(o0).
d4(o1).
d4(o2).
d1(o5).
d2(o5).
d5(o5).
d4(o3).
d2(o0).
d0(o4).

d2(V) :- d0(V).
d3(V) :- d2(V).
d0(V) :- d2(V), not d5(V).
d2(V) :- d4(V).