d1(o4).
d0(o5).
d4(o5).
d5(o4).
d2(o4).
d2(o1).
d3(o5).
d4(o2).
d4(o3).

d4(X) :- d1(X), not d5(X).
d3(X) :- d5(X).
d2(X) :- d0(X).
d1(X) :- not d0(X), d2(X).