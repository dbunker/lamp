d1(o0).
d5(o5).
d4(o5).
d3(o3).
d5(o4).
d3(o0).
d4(o1).
d3(o5).
d5(o3).

d2(X) :- not d5(X), d4(X).
d3(X) :- d5(X), not d4(X).
d1(X) :- d3(X), not d4(X).