d2(o3).
d4(o3).
d5(o7).
d1(o0).
d4(o0).
d1(o9).
d3(o5).
d5(o3).
d3(o2).

d3(X) :- d4(X).
d5(X) :- not d2(X), d4(X).
d1(X) :- d4(X).