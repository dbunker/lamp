d3(o1).
d5(o3).
d1(o5).
d2(o5).
d4(o3).
d0(o3).

d1(X) :- not d0(X), d5(X).
d0(X) :- not d5(X), d1(X).
d5(X) :- d3(X), not d4(X).