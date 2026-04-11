d5(o0).
d0(o0).
d3(o6).
d1(o9).
d3(o3).
d0(o9).
d4(o1).
d1(o3).
d0(o3).

d3(V) :- d1(V).
d4(V) :- d0(V).
d4(V) :- d3(V).
d5(V) :- d0(V), d4(V).
d0(V) :- not d3(V), d4(V).