d2(o3).
d7(o0).
d6(o1).
d4(o5).
d5(o0).
d7(o2).
d9(o2).
d6(o4).
d3(o2).

d8(V) :- d9(V).
d6(V) :- not d3(V), d5(V).
d5(V) :- d3(V).
d5(V) :- d4(V).
d1(V) :- d8(V).
d0(V) :- d2(V).
d0(V) :- d6(V).