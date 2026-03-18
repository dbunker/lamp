d2(o3).
d3(o1).
d3(o3).
d5(o4).
d2(o4).
d3(o5).

d1(X) :- d0(X), d3(X), not d5(X).
d4(X) :- d0(X), d2(X), not d1(X).
d1(X) :- d5(X), not d0(X), d4(X).
d5(X) :- not d2(X), d3(X), d1(X).
d5(X) :- not d2(X), d0(X), d1(X).
d2(X) :- d5(X), not d1(X), d4(X).
d0(X) :- d3(X), d5(X), not d4(X).
d1(X) :- d3(X), not d0(X), d4(X).
d0(X) :- not d2(X), d3(X), d4(X).
d4(X) :- d2(X), d3(X), not d5(X).
d0(X) :- not d3(X), d2(X), d5(X).