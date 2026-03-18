d3(o1).
d3(o2).
d4(o2).
d0(o1).
d0(o2).
d4(o5).
d3(o3).
d1(o2).
d0(o3).

d2(X) :- d4(X), not d0(X), not d1(X).
d5(X) :- not d1(X), d3(X), not d4(X).
d0(X) :- d4(X), not d3(X), not d2(X).
d4(X) :- not d1(X), d0(X), not d3(X).
d3(X) :- not d1(X), d0(X), not d2(X).
d3(X) :- d1(X), not d4(X), not d2(X).
d0(X) :- not d1(X), d5(X), not d4(X).
d5(X) :- d0(X), not d3(X), not d4(X).
d1(X) :- d4(X), not d0(X), not d2(X).
d3(X) :- d0(X), not d4(X), not d2(X).
d0(X) :- d5(X), not d4(X), not d2(X).