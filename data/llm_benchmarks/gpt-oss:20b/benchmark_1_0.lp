d2(o3).
d0(o7).
d9(o8).
d8(o0).
d3(o4).
d9(o7).
d4(o2).
d2(o2).
d7(o7).

d9(V) :- d2(V).
d6(V) :- not d2(V), not d7(V), d9(V).
d0(V) :- not d2(V), d9(V).
d0(V) :- d3(V).
d5(V) :- d2(V).
d1(V) :- not d4(V), not d7(V), d9(V).
d1(V) :- d3(V).